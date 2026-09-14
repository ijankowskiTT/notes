# Notatki o defektach Quasar — #51271 / #56226

## Item 3 — dest stride na unaligned same-width reshard

- branch: `51271-item3-56226-quasar-reshard`, commit `1d9d50df3f3`
- na tym samym shape jest drugi, niezależny bug: #56226

### Problem

- Op zbiera HEIGHT_SHARDED tensor z 4 cores na 2, kopiując row po row. Shape: `(1,1,128,3)` bf16, row-major.
- Każdy row ma tylko 3 elementy bf16, czyli 6 B prawdziwych danych (`unit_size` = packed).
- W L1 ten sam row i tak zajmuje 16 B (`local_unit_size_padded`), bo layout jest alignowany.
- Po skopiowaniu kernel przesuwał dest pointer o packed 6 B zamiast o padded 16 B.
- Kolejny row zaczynał się w środku poprzedniego — dane się nakładały.
- Call wracał bez błędu, output był po prostu zły (silent corruption). Porównanie z torch to łapie; sam „op się skończył” nie.
- Host już liczył te 16 B, ale do kernela podawał tylko `remote_unit_size_padded`. Local dest stride kernel nie znał.
- Na Quasar NOC może czytać od dowolnego bajtu (align 1), ale L1 layout nadal ma 16 B. Factory bierze `buffer()->alignment()` = L1, więc `unaligned` i `+= 6` zostają także tam.

### Podejścia

- **Wybrane:** factory podaje kernelowi `local_unit_size_padded` jako CTA, a kernel po każdym row przesuwa dest o te 16 B. NOC dalej kopiuje tylko packed 6 B. To ten sam pattern co mainline po #53028.
> **CTA (compile-time argument)** — liczba, którą host wkleja w kernel przy JIT, nie przy każdym enqueue. Kernel czyta ją przez `constexpr get_arg`, więc **musi** być CompileTimeArg. `local_unit_size_padded` jest stałe dla tego programu (tu 16), stąd CTA. RTA (runtime argument) to wartości per launch, np. ile readów; w `constexpr get_arg` nie wejdą. Dwa RISC-e na tym samym `.cpp` mogą dostać **różne** CTA (patrz `is_reader` w #56226).
- Nie da się „upchnąć” rows w L1 bez padu. Unpacker i layout L1 wymagają 16 B — tego alignmentu nie ruszamy.
- Nie warto też wysyłać 16 B przez NOC „żeby stride i transfer były równe”. Dest i tak musi skoczyć o 16; bajty padu to śmieci, nie payload.
- Nie skipujemy UNALIGNED na Quasar tylko dlatego, że NOC align = 1. Flaga `unaligned` liczy się z L1 16, nie z NOC. Kernel i tak by robił `+= 6`.
- Host musi liczyć `local_start_offset` tym samym padded stride’em. Inaczej dwa RISC-e dostaną nakładające się dest regiony, nawet gdy każdy z osobna hopa dobrze.

### Rozwiązanie

- Factory wstawia `local_unit_size_padded` do compile-time args kernela. Po każdym skopiowanym row dest pointer skacze o 16 B (początek następnego row w L1).
- NOC nadal przenosi tylko 6 B prawdziwych danych. Padu nie wysyłamy.
- Na hoście `local_start_offset` też idzie padded stride’em, żeby drugi RISC nie zaczął w połowie regionu pierwszego.
- Zmiana w `reshard_same_width_reader.cpp` i `reshard_program_factory_same_width.cpp`.


### Wynik

- Sam ten commit nie naprawia testu: nadal **fail**, bo dwa RISC-e dalej piszą w ten sam scratch (#56226).
- Repro: `test_quasar_reshard_same_width_row_major_local_stride`.

## #56226 — dwa RISC-e, jeden scratch

- twin #51217; mainline fix #53028 (`is_reader` w fold)
- commit `92ced891cef`

### Problem

- Jeden DM core ma **dwa** niezależne RISC-e. Przy tym reshardzie oba dostają ten sam reader kernel.
- Output shard ma 64 rows, podzielone 32 + 32 między te dwa RISC-e.
- Unaligned path najpierw stage’uje remote rows do wspólnego scratch DFB, dopiero potem copy na dest.
- Oba RISC-e piszą pod `scratch + src_offset`. `src_offset` wraca do 0 na każdym nowym remote core, więc drugi nadpisuje pierwszego (clobber).
- Na dest lecą już śmieci ze scratcha — nawet gdy dest stride z item 3 jest poprawne 16 B.
- Na `(1,1,128,3)` widać miss m.in. na `[0,0,31,2]`.
- Quasar Tensix ma te same dwa RISC-e; to nie jest quirk firmware WH.

### Podejścia

- **Wybrane:** podwoić scratch (`num_entries * 2`) i rozdzielić połowy flagą CTA `is_reader`. Tak robi fold / #53028. Nie dokładamy drugiego DFB.
- Osobny DFB na każdy RISC też by rozdzielił zapisy, ale zjada więcej L1 i zmienia ProgramSpec. Nie ten twin.
- Zostawienie tylko jednego RISC-a na unaligned path uniknęłoby race, ale jest wolniejsze i większe niż mainline.
- Serializacja obu RISC-ów na tym samym scratch (jeden czeka, drugi pisze) jest hang-prone i nie ten pattern.
- Sam offset bez podwojenia `num_entries` wychodzi poza scratch (OOB) — drugi RISC pisze w pamięć, której bufor nie ma.

### Rozwiązanie

- Scratch ma `num_entries = 2 * remote_units_per_shard`: miejsce na obie połowy.
- Każdy RISC dostaje CTA `is_reader` = 1 albo 0. Drugi startuje od `remote_units_per_shard * remote_unit_size_padded`, więc pisze obok pierwszego, nie w niego.
- W tym samym commicie są extras, których nie ma w issue (są w #53028). Bez nich unaligned path w ogóle nie wstaje:
  - `#define UNALIGNED` leci zawsze gdy packed != padded, nie tylko gdy `use_scratch`.
  - Writer kernel (`reshard_same_width_writer.cpp`) też hopa padded, gdy local jest input. Nasz test to gather into output, więc oba sloty kompilują **reader** i ta ścieżka nie leci.
  - Borrowed shard DFB: `entry_size = local_unit_size_padded`, a `num_entries` jest clampowane do packed bytes z `TensorSpec`, bo spec-time check nie widzi jeszcze `Buffer`.

> **CTA `is_reader`** — oba RISC-e kompilują ten sam `.cpp`, więc przy JIT dostają różne compile-time args: 1 vs 0. Z tego kernel liczy, która połowa scratcha jest jego. To musi być CTA, bo offset idzie w `constexpr get_arg`; RTA by się tam nie załapało.

### Wynik

- Sam scratch-fix: test nadal **fail**, bo dest hop zostaje 6 B (item 3).
- Sam dest-stride: test nadal **fail**, bo RISC-e dalej clobberują scratch.
- Oba commity razem: **PASS**.
- 2026-09-14 po rebuild: ten test plus 4 stare slice case’y w `test_tm_ops.py` — wszystkie PASSED.
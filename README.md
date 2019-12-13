# MI-PRC

## Použití programu

### Slovníkový útok

**CPU**

Použití: **cesta\_k\_programu 0 cesta\_ke\_slovníku hash** {0-5}[[pravidlo] [abeceda] [minimální délka] [maximální délka]]

**GPU**

Použití: **cesta\_k\_programu 3 cesta\_ke\_slovníku hash** {0-5}[[pravidlo] [abeceda] [minimální délka] [maximální délka]] **počet\_bloků počet\_vláken**

Příklad: ./a.out 3 words.txt bd14a2ab2d51782968669b68b17d909f 1 3 2 2 16 96

### Útok hrubou silou

**CPU**

Použití: **cesta\_k\_programu 1 abeceda hash minimální\_délka maximální\_délka**

Příklad: ./a.out 1 4 fa246d0262c3925617b0c72bb20eeb1d 4 4

**GPU**

Použití: **cesta\_k\_programu 2 abeceda hash minimální\_délka maximální\_délka počet\_bloků počet\_vláken**

---
**abeceda** může nabývat hodnot (0-5):
- 0 – pouze číslice (max 19 znaků)
- 1 – malá písmena (max 13 znaků)
- 2 – velká písmena (max 13 znaků)
- 3 – malá a velká písmena (max 11 znaků)
- 4 – malá a velká písmena, číslice (max 10 znaků)
- 5 – všechny znaky (písmena, číslice, speciální znaky) (max 9 znaků)

**minimimální_délka** řetězce

**mximální_délka** řetězce

**pravidlo** může nabývat hodnotu 1:
- 1 – přidávání řetězců dané abecedy délky minimální – maximální délka za slovo ze slovníku

## Výstup
V případě nalezení řetězce odpovídající zadanému hashi se vypíše daný řetězec, jinak se vypíše „no matches&quot;. Chyby jsou indikované daným výpisem programu.

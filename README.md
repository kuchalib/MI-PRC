# MI-PRC

## Měření

| Heslo  | Spuštění programu                                                                                               | Režim           | Abeceda                    | Délka běhu | Poznámka                                                                          |
|--------|-----------------------------------------------------------------------------------------------------------------|-----------------|----------------------------|------------|-----------------------------------------------------------------------------------|
| hello  | $i = Get-Date>> .\HashSekv.exe 1 1 5d41402abc4b2a76b9719d911017c592 5 5>> $j = Get-Date>> $j-$i                 | Brute force     | malá písmena               | 2283 ms    | 5 znaků                                                                           |
| hello  | $i = Get-Date>> .\HashSekv.exe 0 .\realhuman_phill.txt 5d41402abc4b2a76b9719d911017c592>> $j = Get-Date>> $j-$i | Slovníkový útok | N/A                        | 1126 ms    |                                                                                   |
| zzebra | $i = Get-Date>> .\HashSekv.exe 1 1 d00af4e07a9ff34a790798bc8db5e2fe 6 6>> $j = Get-Date                         | Brute Force     | malá písmena               | 2670 ms    | 6 znaků                                                                           |
| zzebra | $i = Get-Date>> .\HashSekv.exe 0 .\realhuman_phill.txt d00af4e07a9ff34a790798bc8db5e2fe>> $j = Get-Date>> $j-$i | Slovníkový útok | N/A                        | 12510 ms   | Hash nenalezen                                                                    |
| zebra1 | $i = Get-Date>> .\HashSekv.exe 1 4 d85fb95cb761f5874f35ce32c305739b 6 6>> $j = Get-Date>> $j-$i                 | Brute force     | malá a velká písmena čísla | X          | 6 znaků, rekurzivní volání cca 2 hodiny, iterativní 264 hodin!                    |
| 9999   | $i = Get-Date>> .\HashSekv.exe 1 4 fa246d0262c3925617b0c72bb20eeb1d 4 4>> $j = Get-Date>> $j-$i                 | Brute force     | malá a velká písmena čísla | 5106 ms    | 4 znaky, poslední slovo, které se testuje 62**4=14 776 336, cca. 3MHash / sekunda |
| 99999  | $i = Get-Date>> .\HashSekv.exe 1 4 d3eb9a9233e52948740d7eb8c3062d14 5 5>> $j = Get-Date>> $j-$i                 | Brute force     | malá a velká písmena čísla | 317 s      | 5 znaků, poslední slovo které se testuje                                          |
| 99999  | $i = Get-Date>> .\HashSekv.exe 0 .\realhuman_phill.txt d3eb9a9233e52948740d7eb8c3062d14>> $j = Get-Date>> $j-$i | Slovníkový útok | N/A                        | 3308 ms    |                                                                                   |


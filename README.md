# Dosithyroid

Výpočet dávky na štítnou žlázu po aplikaci I-131 pomocí Python aplikace.

## Obsah

- Načtení DICOM souborů ze scintigrafie štítné žlázy po aplikaci I-131.
- Očekává se 6 snímků ve formátu DICOM:
  - Anterior: USW, LSW, PW
  - Posterior: USW, LSW, PW
- Aplikace podporuje:
  - Korekci na mrtvou dobu
  - Automatické zarovnání snímků pomocí konvolučního teorému
  - Segmentaci na 24h snímku, která se následně aplikuje i na ostatní snímky
- Vizuálně jsou zobrazeny pouze PW snímky; USW a LSW jsou zpracovávány na pozadí.
- Výpočet TIAC z planárních snímků s možností doplnění hodnot ze SPECT uptake.
- Po zadání objemu zájmové oblasti jsou vypočteny všechny klíčové dávkové parametry.
- Export klinického protokolu je zatím ve vývoji.

## Autor

**Daniel Ptáček**  
FNKV + ČVUT FJFI

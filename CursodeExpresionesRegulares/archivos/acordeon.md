\w - caracteres de palabras
\d - Dígitos
\s - Espacios/Invisibles en blanco
[0-9] ~ \d
[0-9a-zA-Z_] ~ \w
* greedy - todo
+ uno o más
? cero o uno

## Patrones
mails: `[\w\._]{5,30}\+?[\w]{0,10}@[\w\.\-]{3,}\.\w{2,5}` 
coordenadas 1: `^-?\d{1,3}\.\d{1,6},\s?-?\d{1,3}\.\d{1,6},.*$`
coordenadas 2: `^-?\d{1,2}\s\d{1,2}' \d{1,2}\.\d{2,2}"[WE],\s-?\d{1,2}\s\d{1,2}' \d{1,2}\.\d{2,2}"[NS]$`


^\d+::([\w\s:,\(\)'\.\-&!\/]+)\s\((\d\d\d\d)\)::.*$

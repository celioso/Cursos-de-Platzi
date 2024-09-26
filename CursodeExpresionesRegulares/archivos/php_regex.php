<?php

$file = fopen("results.csv", "r");

$match = 0;
$nomatch = 0;

$t = time();

while(!feof($file)) {
    $line = fgets($file);
    //1913-04-20,France,Luxembourg,8,0,Friendly,Saint-Ouen,France,FALSE
    if (preg_match(
        '/^(\d{4}\-\d\d\-\d\d),(.+),(.+),(\d+),(\d+),.*$/i',
        $line,
        $m
        )
    ) {
        if($m[4] == $m[5]){
            echo "Empate: ";
        } 
        elseif ($m[4] > $m[5]) {
            echo "Local:   ";
        } 
        else {
            echo "Visitante: ";
        }
        printf("\t%s, %s [%d-%d]\n", $m[2], $m[3], $m[4], $m[5]);
        $match++;
    }
    else {
        $nomatch++;
    }
}

fclose($file);

printf("\n\nmatch: %d\nno match: %d\n", $match, $nomatch);

printf("Tiempo: %d\n", time() - $t);
#!/usr/bin/perl

use strict;
use warnings;

my $t = time;

open my $f, "<", "results.csv" or die "No se pudo abrir el archivo: $!";

my $match = 0;
my $nomatch = 0;

while(<$f>) {
    chomp;
    # 2018-06-04,Italy,Netherlands,1,1,Friendly,Turin,Italy,FALSE
    if(m/^([\d]{4,4})\-.*?,(.*?),(.*?),(\d+),(\d+),.*$/) {
        if($5 > $4) {
            printf("%s: %s (%d) - (%d) %s\n",
                $1, $2, $4, $5, $3
            );
        }
        #print $_."\n";
        $match++;
    } 
    else {
        $nomatch++;
    }
}

close($f);

printf("se encontraron: \n - %d matches \n - %d no matches\ntardó %d segundos\n", $match, $nomatch, time() - $t);
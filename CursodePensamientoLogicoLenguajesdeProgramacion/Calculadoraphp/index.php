version corta
<?php
echo '<p>';
function operacione_matematicas(){
  $valor1 = 8;
  $valor2 = 6;
  $suma = $valor1 + $valor2;
  $resta = $valor1 - $valor2;
  $multiplicación = $valor1 * $valor2;
  $división = $valor1 / $valor2;
  echo "La suma es: "; print_r($suma);
  echo '<p>';
  echo "La resta es: "; print_r($resta);
  echo '<p>';
  echo "La multiplicación es: "; print_r($multiplicación);
  echo '<p>';
  echo "La división es: "; print_r($división);
}
operacione_matematicas();
?>
version larga 
<?php
echo '<p>';
function suma(){
  $valor1 = 8;
  $valor2 = 6;
  $resultado = $valor1 + $valor2;
  echo "La suma es "; print_r( $resultado);
  echo '<p>';
}
function resta(){
  $valor1 = 8;
  $valor2 = 6;
  $resultado = $valor1 - $valor2;
  echo "La resta es "; print_r( $resultado);
  echo '<p>';
}

function multiplicación(){
  $valor1 = 8;
  $valor2 = 6;
  $resultado = $valor1 * $valor2;
  echo "La multiplicacion es "; print_r( $resultado);
  echo '<p>';
}
function división(){
  $valor1 = 8;
  $valor2 = 6;
  $resultado = $valor1 / $valor2;
  echo "La division es "; print_r( $resultado);
  echo '<p>';
}

suma();
resta();
multiplicación();
división();
?>
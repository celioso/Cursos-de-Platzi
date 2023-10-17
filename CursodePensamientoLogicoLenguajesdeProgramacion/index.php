<?php
/*tener una lista de ciudades con un clima y una ubicación específica*/

function recomendacion() {
  $clima =array("Bogota" => "frio", "Moteria"=> "calido", "Medellin" => "templado");
  
  $ubicacion = array("guajira" =>"norte", "leticia"=> "sur", "Santander"=> "este", "Antioquia" => "oeste");
  
  $turismo = array("Santa Marta" => "Mar", "Villavicencio" => "Llano", "Riohacha"=> "desierto", "Quindio" => "Valle");

  switch("ubicacion"){
    case "clima":
      echo array_search("calido", $clima);
    break;
    case "ubicacion":
  echo array_search("norte", $ubicacion);
    break;
    case "turismo":
  echo array_search("mar", $turismo);
    break;
    }
}

recomendacion()
?>
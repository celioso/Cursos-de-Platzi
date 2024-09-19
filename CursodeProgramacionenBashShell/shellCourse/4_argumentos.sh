# ! /bin/bash
# Programa para ejemplificar el paos de argumentos

nombreCurso=$1
horarioCurso=$2

echo "El nombre del curso es :$nombreCurso dictado en el horario de $horarioCurso"
echo "El número de parametros enviados es: $#"
echo "Los parametros enviados son: $*"

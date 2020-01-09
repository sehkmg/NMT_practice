#bash bleu.sh HYPO REF

hypo=$1
ref=$2
sed -r 's/(@@ )|(@@ ?$)//g' < $hypo > hypo.tok
sed -r 's/(@@ )|(@@ ?$)//g' < $ref > ref.tok
perl $(dirname "$0")/multi-bleu.perl ref.tok < hypo.tok
rm hypo.tok ref.tok

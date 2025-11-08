make template

for dataset in {0..6}; do
  echo ">> Running on Dataset/${dataset}"
  ./Convolution_template -e "./Convolution/Dataset/${dataset}/output.ppm" \
    -i "./Convolution/Dataset/${dataset}/input0.ppm,./Convolution/Dataset/${dataset}/input1.raw" \
    -o "./test_result_${dataset}.ppm" \
    -t image
  echo
done

dffml accuracy \
  -model pytorchnet \
  -model-features image:int:$((300*300*3)) \
  -model-clstype str \
  -model-classifications rock paper scissors \
  -model-predict label:int:1 \
  -model-network @model.yaml \
  -model-directory rps_model \
  -model-imageSize 150 \
  -model-enableGPU \
  -sources f=dir \
    -source-foldername rps-test-set \
    -source-feature image \
    -source-labels rock paper scissors \
  -scorer pytorchscore
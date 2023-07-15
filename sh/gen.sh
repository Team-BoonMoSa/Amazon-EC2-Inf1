python3 server/model-repository/gen_triton_model.py \
    --model_type pytorch \
    --triton_input images,TYPE_FP32,1x3x640x640 \
    --triton_output output0,TYPE_FP32,1x25200x38 output1,TYPE_FP32,1x32x160x160 \
    --compiled_model server/model-repository/BoonMoSa/1/model_neuron.pt \
    --neuron_core_range 0:3 \
    --triton_model_dir BoonMoSa
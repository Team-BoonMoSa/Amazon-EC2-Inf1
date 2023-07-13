import triton_python_backend_utils as pb_utils
import torch


class TritonPythonModel:
    def initialize(self, pbtxt_args):
        self.model_config = json.loads(pbtxt_args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "output0"
        )
        output1_config = pb_utils.get_output_config_by_name(
            self.model_config, "output1"
        )

        self.output_dtype_img = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        self.output_dtype_link = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )

        self.model = torch.jit.load('model_neuron.pt')

    def execute(self, requests):
        response = []
        for request in requests:
            images = pb_utils.get_input_tensor_by_name(request, "images")
            output = self.model(images.as_numpy())

            output0 = pb_utils.Tensor(
                "output0", np.array(output).astype(self.output_dtype_img)
            )

            output1 = pb_utils.Tensor(
                "output1", np.array(output).astype(self.output_dtype_img)
            )

            inference_response = pb_utils.InferenceResponse(output_tensors=[output0, output1])
            responses.append(inference_response)

use orion::operators::tensor::{Tensor, TensorTrait};
use orion::operators::tensor::{U32Tensor, I32Tensor, I8Tensor, FP8x23Tensor, FP16x16Tensor, FP32x32Tensor, BoolTensor};
use orion::numbers::{FP8x23, FP16x16, FP32x32};
use orion::operators::nn::{NNTrait, FP16x16NN};

use node_linear_weight::get_node_linear_weight;
use node_linear_bias::get_node_linear_bias;

fn main(node_input: Tensor<FP16x16>) -> Tensor<FP16x16> {
let node_output = NNTrait::gemm(node_input, get_node_linear_weight(), Option::Some(get_node_linear_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);

       return node_output;
    }
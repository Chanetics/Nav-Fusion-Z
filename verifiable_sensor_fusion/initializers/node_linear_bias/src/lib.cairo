use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_linear_bias() -> Tensor<FP16x16> {
    let mut shape = array![1];

    let mut data = array![FP16x16 { mag: 6351, sign: false }];

    TensorTrait::new(shape.span(), data.span())
}
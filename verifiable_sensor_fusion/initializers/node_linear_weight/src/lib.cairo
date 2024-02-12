use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_linear_weight() -> Tensor<FP16x16> {
    let mut shape = array![1, 10];

    let mut data = array![FP16x16 { mag: 2116, sign: true }, FP16x16 { mag: 15882, sign: true }, FP16x16 { mag: 8533, sign: false }, FP16x16 { mag: 2174, sign: false }, FP16x16 { mag: 1224, sign: false }, FP16x16 { mag: 3620, sign: true }, FP16x16 { mag: 11969, sign: true }, FP16x16 { mag: 18679, sign: true }, FP16x16 { mag: 93906, sign: false }, FP16x16 { mag: 2631, sign: false }];

    TensorTrait::new(shape.span(), data.span())
}
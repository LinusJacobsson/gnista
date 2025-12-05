pub struct Tensor {
    data: Vec<f32>,
    grads: Option<Vec<f32>>,
    shape: Vec<usize>,
    requires_grad: bool,
}


fn main() {
    let tensor = Tensor {
        data: vec![1.0, 2.0, 3.0],
        grads: None,
        shape: vec![3],
        requires_grad: true,
    };
}
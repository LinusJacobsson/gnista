use std::ops::Add;
use std::ops::Sub;
#[derive(Debug)]
pub struct Tensor {
    data: Vec<f32>,
    grads: Option<Vec<f32>>,
    shape: Vec<usize>,
    requires_grad: bool,
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Tensor {
        if self.shape != other.shape {
            panic!("Shapes do not match for addition");
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Tensor {
            data,
            grads: None,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad || other.requires_grad,
        }
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, other: Tensor) -> Tensor {
        if self.shape != other.shape {
            panic!("Shapes do not match for subtraction");
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Tensor {
            data,
            grads: None,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad || other.requires_grad,
        }
    }
}
impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        Tensor {
            data,
            grads: None,
            shape,
            requires_grad,
        }
    }

    pub fn data(&self) -> &Vec<f32> {
        &self.data
    }
}

fn main() {
    let tensor = Tensor {
        data: vec![1.0, 2.0, 3.0, 4.0],
        grads: None,
        shape: vec![3],
        requires_grad: false,
    };

    let another_tensor = Tensor {
        data: vec![4.0, 5.0, 6.0, 7.0],
        grads: None,
        shape: vec![3],
        requires_grad: false,
    };

    println!("Tensor 1 data: {:?}", tensor);
    println!("Tensor 2 data: {:?}", another_tensor);
    let result_tensor = tensor + another_tensor;
    println!("Result Tensor data: {:?}", result_tensor);
}

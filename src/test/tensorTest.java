package test;

import deepj.operations.Add;
import deepj.operations.Divide;
import deepj.operations.Multiply;
import deepj.tensors.Tensor;
import deepj.tensors.TensorUtils;

import java.util.Arrays;

public class tensorTest {
    public static void main(String[] args) {
//        Tensor t = TensorUtils.ones(1);
//        Graph g = t.add(TensorUtils.ones(1)).compile();
//
//        t = t.sub(g.result().info());
        Tensor t1 = TensorUtils.ones(10, 10);
        Tensor t2 = TensorUtils.fill(100, 10, 10);

        Tensor t3 = new Divide(t1, t2);

        System.out.println(Arrays.toString(t3.get()));
    }
}

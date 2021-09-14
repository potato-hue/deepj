package test;

import deepj.computation.Graph;
import deepj.tensors.*;

import java.util.Arrays;

public class GraphTest {
    public static void main(String[] args) {
        Tensor t1 = TensorUtils.ones(10, 10);
        Tensor t2 = TensorUtils.fill(100, 10, 10);

        t1.requireGrads(true);

        for (int i = 0; i < 100; i++) {
            Tensor t = t1.add(t2);
            Graph g = t.construct();
//            g.getGradOf(t1).get();
            System.out.println(g.getGradOf(t1));
            t1 = t1.sub(g.getGradOf(t1));
            t1.requireGrads(true);
        }
    }
}

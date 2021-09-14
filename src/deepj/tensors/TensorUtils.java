package deepj.tensors;

import java.util.Arrays;

public class TensorUtils {
    private TensorUtils(){}

    /**
     * returns a tensor filled with 0 with the specified shape
     * @param shape the shape of the tensor to be returned
     * @return the tensor filled with 0's
     */
    public static CPUTensor zeroes(int... shape){
        int total = Arrays.stream(shape).reduce(1, (a, b)->a * b);
        double[]zeroes = new double[total];
        Arrays.fill(zeroes, 0d);

        return  new CPUTensor(zeroes, shape);
    }
    /**
     * returns a tensor filled with 1 with the specified shape
     * @param shape the shape of the tensor to be returned
     * @return the tensor filled with 1's
     */
    public static CPUTensor ones(int... shape){
        int total = Arrays.stream(shape).reduce(1, (a, b)->a * b);
        double[]ones = new double[total];
        Arrays.fill(ones, 1d);

        return new CPUTensor(ones, shape);
    }
    public static CPUTensor fill(double val, int... shape){
        int total = Arrays.stream(shape).reduce(1, (a, b)->a * b);
        double[]values = new double[total];
        Arrays.fill(values, val);

        return new CPUTensor(values, shape);
    }
    public static Tensor create(double[]values, int... shape){
        return new CPUTensor(values, shape);
    }
}

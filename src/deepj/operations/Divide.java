package deepj.operations;

import deepj.tensors.Tensor;
import deepj.tensors.TensorUtils;

public class Divide extends Operation{
    private Tensor divisor, dividend;

    public Divide(Tensor dividend, Tensor divisor) {
        super();
        this.divisor = divisor;
        this.dividend = dividend;
        setAllPrev(dividend, divisor);
    }

    @Override
    public void compile() {
        int len = divisor.get().length;
        double[]values = new double[len];

        for (int i = 0; i < len; i++){
            values[i] = dividend.get()[i]/divisor.get()[i];
        }
        setValue(TensorUtils.create(values, divisor.getShape()));
    }
}

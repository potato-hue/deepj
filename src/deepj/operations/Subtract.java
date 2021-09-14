package deepj.operations;

import deepj.tensors.Tensor;
import deepj.tensors.TensorUtils;

public class Subtract extends Operation{
    private Tensor subtrahend, minuend;

    public Subtract(Tensor minuend, Tensor subtrahend) {
        super();
        this.subtrahend = subtrahend;
        this.minuend = minuend;
        setAllPrev(subtrahend, minuend);
    }

    @Override
    public void compile() {
        int len = subtrahend.get().length;
        double[]values = new double[len];

        for (int i = 0; i < len; i++){
            values[i] = minuend.get()[i] - subtrahend.get()[i];
        }
        setValue(TensorUtils.create(values, subtrahend.getShape()));
    }
}

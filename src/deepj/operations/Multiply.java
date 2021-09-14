package deepj.operations;

import deepj.tensors.Tensor;
import deepj.tensors.TensorUtils;

public class Multiply extends Operation{
    private final Tensor[]allTensors;
    public Multiply(Tensor... tensors){
        super();
        this.allTensors = tensors;
        setAllPrev(tensors);
    }

    @Override
    public void compile() {
        int len = allTensors[0].get().length;
        double[]values = new double[len];


        for (int i = 0; i < len; i++) {
            values[i] = 1;
            for (Tensor allTensor : allTensors) {
                values[i] *= allTensor.get()[i];
            }
        }
        Tensor result = TensorUtils.create(values, allTensors[0].getShape());
        setValue(result);
        for(Tensor t : allTensors){
            if(t.requiresGrads())setGrad(t, result.divide(t));
        }
    }
}

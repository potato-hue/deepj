package deepj.operations;

import deepj.tensors.Tensor;
import deepj.tensors.TensorUtils;

public class Add extends Operation{
    private final Tensor[]allTensors;
    public Add(Tensor... tensors){
        super();
        this.allTensors = tensors;
        setAllPrev(tensors);
        for(Tensor t : tensors){
            if(t.requiresGrads()){
                setGrad(t, TensorUtils.ones(t.getShape()));
            }
        }
    }

    @Override
    public void compile() {
        int len = allTensors[0].get().length;
        double[]values = new double[len];

        for (int i = 0; i < len; i++) {
            for (Tensor allTensor : allTensors) {
                values[i] += allTensor.get()[i];
            }
        }
        setValue(TensorUtils.create(values, allTensors[0].getShape()));
    }
}

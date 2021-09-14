package deepj.tensors;

import deepj.computation.Grad;
import deepj.computation.Graph;
import deepj.operations.Add;
import deepj.operations.Divide;
import deepj.operations.Multiply;
import deepj.operations.Subtract;

import java.util.Arrays;

public interface Tensor {
    default Tensor getTensor() {
        return null;
    }

    default Tensor get(int... index) {
        return getTensor().get(index);
    }

    default double getValue(int... index) {
        return getTensor().getValue(index);
    }

    default double[] get() {
        return getTensor().get();
    }
    default int[] getShape(){
        return getTensor().getShape();
    }
    default Tensor add(Tensor... tensors){
        Tensor[]all = Arrays.copyOf(tensors, tensors.length + 1);
        all[tensors.length] = this;
        return new Add(all);
    }
    default Tensor mul(Tensor... tensors){
        Tensor[]all = Arrays.copyOf(tensors, tensors.length + 1);
        all[tensors.length] = this;
        return new Multiply(all);
    }

    default void computeGrads(Grad grad){
        getTensor().computeGrads(grad);
    }
    default void setGraph(Graph g){
        getTensor().setGraph(g);
    }
    default boolean requiresGrads(){
        return getTensor().requiresGrads();
    }
    default Tensor divide(Tensor divisor){
        return new Divide(this, divisor);
    }
    default void requireGrads(boolean val){
        getTensor().requireGrads(val);
    };
    default Graph construct(){
        Graph g = new Graph(this);
        g.backwards();
        g.computeGrads();
        return g;
    }
    default Tensor sub(Tensor t){
        return new Subtract(this, t);
    }
}

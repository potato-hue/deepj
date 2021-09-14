package deepj.operations;

import deepj.computation.Grad;
import deepj.computation.Graph;
import deepj.tensors.Tensor;

import java.util.HashMap;

public abstract class Operation implements Tensor {
    private boolean compiled;
    private boolean requiresGrads;
    Tensor value;
    Tensor[]allPrev;
    Graph g;
    HashMap<Tensor, Tensor>gradients;
    public Operation(){
        gradients = new HashMap<>();
    }
    @Override
    public Tensor getTensor(){
        if(!compiled)compile();
        return value;
    }
    public boolean  compiled(){
        return compiled;
    }

    private void setCompiled(boolean compiled){
        this.compiled = compiled;
    }
    public abstract void compile();
    public void setValue(Tensor t){
        this.value = t;
    }
    public void computeGrads(Grad grad){
        for (Tensor t : allPrev){
            if(t.requiresGrads())
            g.pushGrad(this, t, gradients.get(t).mul(grad.getValue()));
        }
    }
    public void setAllPrev(Tensor... allPrev){
        this.allPrev = allPrev;
    }
    public void setGraph(Graph g){
        this.g = g;
        for(Tensor t : allPrev){
            t.setGraph(g);
            if(t.requiresGrads())g.addRecipient(this, t);
        }
    }
    public boolean requiresGrads(){
        if(requiresGrads)return true;
        for (Tensor t : allPrev)if(t.requiresGrads())return true;
        return false;
    }
    public void setRecipient(Tensor receiver){
        g.addRecipient(this, receiver);
    }
    public void pushGrad(Tensor receiver, Tensor value){
        g.pushGrad(this, receiver, value);
    }
    public void setGrad(Tensor t, Tensor grad){
        this.gradients.put(t, grad);
    }
    public void requireGrads(boolean val){
        this.requiresGrads = true;
    }
}

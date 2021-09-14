package deepj.computation;

import deepj.tensors.Tensor;

import java.util.HashMap;

public class AutoGrad {
    private final HashMap<Tensor, Grad>grads;

    public AutoGrad(){
        grads = new HashMap<>();
    }
    public void root(Tensor tensor){
        checkExists(tensor);
        grads.get(tensor).setRoot();
    }
    public void addRecipient(Tensor sender, Tensor receiver){
        checkExists(receiver);
        grads.get(receiver).add(sender);
    }

    public void pushGrad(Tensor sender, Tensor receiver, Tensor value){
        grads.get(receiver).push(value, sender);
    }

    public void checkExists(Tensor t){
        if(grads.containsKey(t))return;
        grads.put(t, new Grad(t));
    }

    public Grad gradOf(Tensor node) {
        return grads.get(node);
    }
}

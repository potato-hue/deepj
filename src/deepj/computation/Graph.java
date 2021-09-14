package deepj.computation;

import deepj.tensors.Tensor;

public class Graph {
    AutoGrad gradSys;
    Tensor rootNode;

    public Graph(Tensor rootNode){
            this.rootNode = rootNode;
            gradSys = new AutoGrad();
    }

    public void backwards(){
        rootNode.setGraph(this);

    }
    public void computeGrads(){
        gradSys.root(rootNode);
        rootNode.computeGrads(gradSys.gradOf(rootNode));
    }
    public void addRecipient(Tensor sender, Tensor recipient){
        gradSys.addRecipient(sender, recipient);
    }
    public void pushGrad(Tensor sender, Tensor recipient, Tensor value){
        gradSys.pushGrad(sender, recipient, value);
    }
    public Tensor getGradOf(Tensor t){
        return gradSys.gradOf(t).value;
    }
}

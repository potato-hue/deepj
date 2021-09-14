
package deepj.computation;

        import deepj.tensors.Tensor;
        import deepj.tensors.TensorUtils;

        import java.util.ArrayList;

public class Grad{
    ArrayList<Tensor>senders;
    private final Tensor recipient;
    Tensor value;

    public Grad(Tensor t){
        this.recipient = t;
        value = TensorUtils.zeroes(t.getShape());
        senders = new ArrayList<>();

    }

    public void push(Tensor value, Tensor sender){
        senders.remove(sender);
        this.value = this.value.add(value);

        if(senders.isEmpty()){
            recipient.computeGrads(this);
        }

    }
    public Tensor getValue(){
        return value;
    }
    public void add(Tensor sender){
        this.senders.add(sender);
    }
    public void setRoot(){
        value = TensorUtils.ones(value.getShape());
    }
}
package deepj.tensors;

import deepj.computation.Grad;
import deepj.computation.Graph;

import java.util.Arrays;

public class CPUTensor implements Tensor{
    private final double[]values;
    private final int[]shape;
    private final int numElements;
    private Graph g;
    private boolean requiresGrads;

    /**
    * Get the tensor at the following index
    * @param index the index to search the tensor for
    * @return the tensor at the following index
    *
    */
    public CPUTensor get(int... index) {
        // check for dimension validity
        if(index.length > shape.length){
            throw new IllegalArgumentException(String.format("invalid dimensions, expected index of length <= %d instead got value of %d", shape.length, index.length));
        }
        int startIndex = 0;
        int left = numElements;
        for (int i = 0; i < index.length; i++) {
            // check of index validity
            if(index[i] >= shape[i])throw new IllegalArgumentException("cannot give index for more than shape");
            startIndex += index[i] * (left = left/shape[i]);
        }
        int endIndex = startIndex;
        for (int i = index.length; i < shape.length; i++) {
            endIndex += ((shape[i] - 1) * (left = left/shape[i])) ;
        }
        return new CPUTensor(Arrays.copyOfRange(values, startIndex, endIndex + 1), Arrays.copyOfRange(shape, index.length, shape.length));
    }

    public boolean requiresGrads(){
        return requiresGrads;
    }
    @Override
    public double[] get() {
        return this.values.clone();
    }

    /**
     * The most to be used constructor of Tensor
     * @param value the values of the newly created Tensor
     * @param shape the shape of the tensor
     */

    public CPUTensor(double[]value, int... shape){
        // match the length of the array given to the shape
        int totalValue = Arrays.stream(shape).reduce(1, (a, b) -> a * b);

        // if value does not match throw an exception
        if(totalValue != value.length)throw new IllegalArgumentException(String.format("expected values of length %d instead got values of length %d", totalValue, value.length));

        // set attributes
        this.shape = shape;
        this.values = value.clone();
        this.numElements = totalValue;
    }

    /**
     * returns the string version of the tensor
     * @return the string version of the tensor
     */
    @Override
    public String toString(){
        return Arrays.toString(values);
    }
    public double getValue(int... index){
        if(index.length != shape.length){
            throw new IllegalArgumentException(String.format("invalid dimensions, expected index of length equal to %d instead got value of %d", shape.length, index.length));
        }
        int position = 0;
        int left = numElements;
        for (int i = 0; i < index.length; i++) {
            if(index[i] < 0){
                throw new IllegalArgumentException("negative indices");
            }
            position = index[i] * (left = left/shape[i]);
        }
        return values[position];
    }


    /**
     * get the shape of the tensor
     * @return the shape of the tensor
     */
    @Override
    public int[]getShape(){
        return shape.clone();
    }

    /**
     * get the value at the linear buffer
     * @param index the index in linear
     * @return the value at the linear index
     */
    public double getLinear(int index){
        return values[index];
    }
    public void computeGrads(Grad grad){

    }
    public void setGraph(Graph graph){
        this.g = graph;
    }
    public void requireGrads(boolean val){
        this.requiresGrads = val;
    }
}

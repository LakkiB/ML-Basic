package cs475;

public class Base
{
    public static void main(String[] args)
    {
        Base test =  new Child();

        test.test();
    }

    protected void test ()
    {
        System.out.println("parent");
    }
}

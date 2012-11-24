package cs475.classifier.unsupervisedlearning.image_denoise;


import java.util.*;

public class MRFImageProcessor
{
    public MRFImageProcessor (
            double eta,
            double beta,
            double omega,
            int num_iterations,
            boolean use_second_level,
            int num_k )
    {
        this.eta = eta;
        this.beta = beta;
        this.omega = omega;
        this.iterations = num_iterations;
        this.useSecondLevel = use_second_level;
        this.numK = num_k;

    }

    public int[][] denoisifyImage ( int[][] observedImage )
    {
        switch ( ImageUtils.countColors( observedImage, false ) )
        {
            case 1:
                return observedImage;

            case 2:
                return denoisifyBlackAndWhiteImage( observedImage, useSecondLevel );

            default:
                return denoisifyGreyScaleImage( observedImage, useSecondLevel );
        }
    }

    private int[][] denoisifyGreyScaleImage ( int[][] observedImage, boolean useSecondLevel )
    {
        if ( useSecondLevel )
        {
            return denoisifyGSImageWith2HiddenLevels( observedImage );
        }
        else
        {
            return denoisifyGreyScaleImage( observedImage );
        }
    }

    private int[][] denoisifyBlackAndWhiteImage ( int[][] observedImage, boolean useSecondLevel )
    {
        if ( useSecondLevel )
        {
            return denoisifyBWImageWith2HiddenLevels( observedImage );
        }
        else
        {
            return denoisifyBlackAndWhiteImage( observedImage );
        }
    }


    private void initializeHiddenNodesIn2ndLevel ( double[][] zLayer, int[][] xLayer )
    {
        for ( int ii = 0 ; ii < xLayer.length ; ii++ )
        {
            for ( int jj = 0 ; jj < xLayer[ ii ].length ; jj++ )
            {
                int m = ( int ) Math.floor( ii / numK );
                int n = ( int ) Math.floor( jj / numK );
                zLayer[ m ][ n ] += xLayer[ ii ][ jj ];
            }
        }

        for ( int ii = 0 ; ii < zLayer.length ; ii++ )
        {
            for ( int jj = 0 ; jj < zLayer[ ii ].length ; jj++ )
            {
                zLayer[ ii ][ jj ] /= Math.pow( numK, 2 );
            }
        }
    }


    private boolean isConnected ( int i, int j, int m, int n, int numK )
    {
        return ( Math.floor( i / numK ) == m ) && ( Math.floor( j / numK ) == n );
    }


    private int[][] denoisifyGSImageWith2HiddenLevels ( int[][] yLayer )
    {
        int[][] xLayer = new int[ yLayer.length ][];
        double[][] zLayer = new double[ yLayer.length / numK ][];

        initializeHiddenNodes( xLayer, yLayer );
        initializeHiddenNodesIn2ndLevel( zLayer, xLayer );
        HashMap<Integer, Integer> colorMap = ImageUtils.createColorMap( yLayer );

        while ( iterations-- > 0 )
        {
            for ( int ii = 0 ; ii < xLayer.length ; ii++ )
            {
                for ( int jj = 0 ; jj < xLayer[ ii ].length ; jj++ )
                {
                    List<Integer> neighbors = getNeighbors( xLayer, ii, jj );
                    int finalColorValue = decideColorGreyScaleEx( yLayer[ ii ][ jj ],
                            zLayer[ ii / numK ][ jj / numK ], colorMap, neighbors );
                    xLayer[ ii ][ jj ] = finalColorValue;
                }
            }
        }
        return xLayer;
    }

    private int[][] denoisifyBWImageWith2HiddenLevels ( int[][] yLayer )
    {
        int[][] xLayer = new int[ yLayer.length ][];
        double[][] zLayer = new double[ yLayer.length / numK ][];

        initializeHiddenNodes( xLayer, yLayer );
        initializeHiddenNodesIn2ndLevel( zLayer, xLayer );
        HashMap<Integer, Integer> colorMap = ImageUtils.createColorMap( yLayer );

        while ( iterations-- > 0 )
        {
            for ( int ii = 0 ; ii < xLayer.length ; ii++ )
            {
                for ( int jj = 0 ; jj < xLayer[ ii ].length ; jj++ )
                {
                    List<Integer> neighbors = getNeighbors( xLayer, ii, jj );
                    int finalColorValue = decideColorBWEx( yLayer[ ii ][ jj ],
                            zLayer[ ii / numK ][ jj / numK ], colorMap, neighbors );
                    xLayer[ ii ][ jj ] = finalColorValue;
                }
            }
        }
        return xLayer;

    }

    private int[][] denoisifyGreyScaleImage ( int[][] observedImage )
    {
        int[][] hiddenNodes = new int[ observedImage.length ][];
        initializeHiddenNodes( hiddenNodes, observedImage );
        HashMap<Integer, Integer> colorMap = ImageUtils.createColorMap( observedImage );

        while ( iterations-- > 0 )
        {
            for ( int ii = 0 ; ii < hiddenNodes.length ; ii++ )
            {
                for ( int jj = 0 ; jj < hiddenNodes[ ii ].length ; jj++ )
                {
                    List<Integer> neighbors = getNeighbors( hiddenNodes, ii, jj );
                    int finalColorValue = decideColorGreyScale( observedImage[ ii ][ jj ], colorMap, neighbors );
                    hiddenNodes[ ii ][ jj ] = finalColorValue;
                }
            }
        }
        return hiddenNodes;
    }

    private int[][] denoisifyBlackAndWhiteImage ( int[][] observedNodes )
    {
        int[][] hiddenNodes = new int[ observedNodes.length ][];
        initializeHiddenNodes( hiddenNodes, observedNodes );
        HashMap<Integer, Integer> colorMap = ImageUtils.createColorMap( observedNodes );

        while ( iterations-- > 0 )
        {
            for ( int ii = 0 ; ii < hiddenNodes.length ; ii++ )
            {
                for ( int jj = 0 ; jj < hiddenNodes[ ii ].length ; jj++ )
                {
                    List<Integer> neighbors = getNeighbors( hiddenNodes, ii, jj );
                    int finalState = decideColorBorW( observedNodes[ ii ][ jj ], colorMap, neighbors );
                    hiddenNodes[ ii ][ jj ] = finalState;
                }
            }
        }
        return hiddenNodes;
    }

    private int decideColorBorW ( int observedNode, HashMap<Integer, Integer> colorMap, List<Integer> neighbors )
    {
        SortedMap<Double, Integer> correlations = new TreeMap<Double, Integer>();

        for ( int color : colorMap.keySet() )
        {
            double correlation = computeJointProbabilityForBAndWPixel( observedNode, colorMap.get( color ), neighbors );
            correlations.put( correlation, colorMap.get( color ) );
        }

        return correlations.get( correlations.lastKey() );
    }


    private int decideColorBWEx (
            int observedNode,
            double hiddenLayer2NodeValue,
            HashMap<Integer, Integer> colorMap,
            List<Integer> neighbors )
    {
        SortedMap<Double, Integer> correlations = new TreeMap<Double, Integer>();

        for ( int color : colorMap.keySet() )
        {
            double correlation = computeJointProbabilityForBAndWPixelEx( observedNode, colorMap.get( color ),
                    hiddenLayer2NodeValue, neighbors );
            correlations.put( correlation, colorMap.get( color ) );
        }

        return correlations.get( correlations.lastKey() );
    }


    private int decideColorGreyScale ( int observedNode, HashMap<Integer, Integer> colorMap, List<Integer> neighbors )
    {
        SortedMap<Double, Integer> correlations = new TreeMap<Double, Integer>();

        for ( int color : colorMap.keySet() )
        {
            double correlation = computeJointProbabilityForGreyScalePixel( observedNode, colorMap.get( color ),
                    neighbors );
            correlations.put( correlation, colorMap.get( color ) );
        }

        return correlations.get( correlations.lastKey() );
    }


    private int decideColorGreyScaleEx (
            int observedNode,
            double hiddenLayer2Node,
            HashMap<Integer, Integer> colorMap,
            List<Integer> neighbors )
    {
        SortedMap<Double, Integer> correlations = new TreeMap<Double, Integer>();

        for ( int color : colorMap.keySet() )
        {
            double correlation = computeJointProbabilityForGreyScalePixelEx
                    ( observedNode, colorMap.get( color ), hiddenLayer2Node, neighbors );

            correlations.put( correlation, colorMap.get( color ) );
        }

        return correlations.get( correlations.lastKey() );
    }


    private double computeJointProbabilityForGreyScalePixel (
            int observedNode,
            int hiddenNodeValue,
            List<Integer> hiddenNeighbors )
    {
        double potentialXiXj = 0, potentialXiYi = 0;

        for ( Integer neighbor : hiddenNeighbors )
        {
            potentialXiXj += ( Math.log( Math.abs( hiddenNodeValue - hiddenNeighbors.get( neighbor ) ) + 1 ) - 1 ) *
                    beta;
        }
        potentialXiYi = ( ( Math.log( Math.abs( hiddenNodeValue - observedNode ) ) + 1 ) - 1 ) * eta;

        double energy = -potentialXiXj - potentialXiYi;
        return Math.exp( -energy );
    }


    private double computeJointProbabilityForGreyScalePixelEx (
            int observedNode,
            double hiddenNodeValue,
            double hiddenLayer2NodeValue,
            List<Integer> hiddenNeighbors )
    {
        double potentialXiXj = 0, potentialXiYi, potentialXiZj;

        for ( Integer hiddenNeighbor : hiddenNeighbors )
        {
            potentialXiXj += ( Math.log( Math.abs( hiddenNodeValue - hiddenNeighbor ) + 1 ) - 1 ) * beta;
        }

        potentialXiYi = ( ( Math.log( Math.abs( hiddenNodeValue - observedNode ) ) + 1 ) - 1 ) * eta;
        potentialXiZj = ( ( Math.log( Math.abs( hiddenNodeValue - hiddenLayer2NodeValue ) ) + 1 ) - 1 ) * omega;

        double energy = -potentialXiXj - potentialXiYi - potentialXiZj;
        return Math.exp( -energy );
    }


    private double computeJointProbabilityForBAndWPixel (
            int observedNode,
            int hiddenNodeValue,
            List<Integer> hiddenNeighbors )
    {
        double summationXiXj = 0;
        for ( Integer neighbor : hiddenNeighbors )
        {
            summationXiXj += hiddenNodeValue * neighbor;
        }
        double energy = -( beta * summationXiXj ) - ( eta * hiddenNodeValue * observedNode );
        return Math.exp( -energy );
    }


    private double computeJointProbabilityForBAndWPixelEx (
            int observedNode,
            int hiddenNodeValue,
            double hiddenLayer2NodeValue,
            List<Integer> hiddenNeighbors )
    {
        double summationXiXj = 0;
        for ( Integer neighbor : hiddenNeighbors )
        {
            summationXiXj += hiddenNodeValue * neighbor;
        }
        double energy = -( beta * summationXiXj ) - ( eta * hiddenNodeValue * observedNode ) - ( omega *
                hiddenNodeValue * hiddenLayer2NodeValue );

        return Math.exp( -energy );
    }


    private List<Integer> getNeighbors ( int[][] hiddenNodes, int rowIdx, int colIdx )
    {
        List<Integer> neighbors = new ArrayList<Integer>();
        neighbors.add( hiddenNodes[ rowIdx ][ colIdx - 1 ] );  // left neighbor
        neighbors.add( hiddenNodes[ rowIdx ][ colIdx + 1 ] );  // right neighbor
        neighbors.add( hiddenNodes[ rowIdx - 1 ][ colIdx ] );  // top neighbor
        neighbors.add( hiddenNodes[ rowIdx + 1 ][ colIdx ] );  // bottom neighbor
        return neighbors;
    }


    private void initializeHiddenNodes ( int[][] hiddenNodes, int[][] observedNodes )
    {
        for ( int ii = 0 ; ii < observedNodes.length ; ii++ )
        {
            for ( int jj = 0 ; jj < observedNodes[ ii ].length ; jj++ )
            {
                hiddenNodes[ ii ][ jj ] = observedNodes[ ii ][ jj ];
            }
        }
    }


    private double eta;
    private double beta;
    private double omega;

    private int numK;
    private int iterations;

    private boolean useSecondLevel;
}

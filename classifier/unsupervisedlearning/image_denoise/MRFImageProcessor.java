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
        this.eta            = eta;
        this.beta           = beta;
        this.numK           = num_k;
        this.omega          = omega;
        this.iterations     = num_iterations;
        this.useSecondLevel = use_second_level;

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


    private void initializeHiddenNodesIn2ndLevel (
            double[][] zLayer,
            int[][] xLayer,
            HashMap<Integer, Integer> colorMap )
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

        if ( colorMap.size() == 2 )
        {
            if(colorsAvg == 0)
            {
                initColorStatsForBWClassification( colorMap );
            }

            for ( int ii = 0 ; ii < zLayer.length ; ii++ )
            {
                for ( int jj = 0 ; jj < zLayer[ ii ].length ; jj++ )
                {
                    double avg = zLayer[ ii ][ jj ] / Math.pow( numK, 2 );
                    zLayer[ ii ][ jj ] = avg >= colorsAvg ? maxColor : minColor;
                }
            }
        }
        else
        {
            int rLimit = xLayer.length - 1;
            int cLimit = xLayer[0].length - 1;

            for ( int ii = 0 ; ii < zLayer.length ; ii++ )
            {
                for ( int jj = 0 ; jj < zLayer[ ii ].length ; jj++ )
                {
                    int rDen, cDen;
                    rDen = rLimit - ii * numK >= numK ? numK : numK - ii % numK;
                    cDen = cLimit - jj * numK >= numK ? numK : numK - jj % numK;

                    zLayer[ ii ][ jj ] /= ( rDen * cDen );
                }
            }
        }
    }

    private void initColorStatsForBWClassification ( HashMap<Integer, Integer> colorMap )
    {
        colorsAvg = getColorAverage( colorMap );
        maxColor = getMaxColor( colorMap );
        minColor = getMinColor( colorMap );
    }

    private double getMaxColor ( HashMap<Integer, Integer> colorMap )
    {
        double max = Double.MIN_VALUE;
        for(int color : colorMap.values())
        {
            if(color > max )
                max = color;
        }
        return max;
    }


    private double getMinColor ( HashMap<Integer, Integer> colorMap )
    {
        double min = Double.MAX_VALUE;
        for(int color : colorMap.values())
        {
            if(color <= min )
                min = color;
        }
        return min;
    }

    private double getColorAverage ( HashMap<Integer, Integer> colorMap )
    {
        double colAvg = 0;
        for (Integer col : colorMap.values())
        {
            colAvg += col;
        }
        colAvg /= colorMap.size();
        return colAvg;
    }


    private boolean isConnected ( int i, int j, int m, int n, int numK )
    {
        return ( Math.floor( i / numK ) == m ) && ( Math.floor( j / numK ) == n );
    }


    private int[][] denoisifyGSImageWith2HiddenLevels ( int[][] yLayer )
    {
        int[][] xLayer = new int[ yLayer.length ][ yLayer[ 0 ].length ];
        int row = ( yLayer.length / numK == yLayer.length / numK + yLayer.length % numK ) ? yLayer.length / numK :
                yLayer.length / numK + 1;
        int col = ( yLayer[ 0 ].length / numK == yLayer[ 0 ].length / numK + yLayer[ 0 ].length % numK ) ? yLayer[ 0
                ].length / numK : yLayer[ 0 ].length / numK + 1;

        double[][] zLayer = new double[ row ][ col ];
        HashMap<Integer, Integer> colorMap = ImageUtils.createColorMap( yLayer );

        initializeHiddenNodes( xLayer, yLayer );
        initializeHiddenNodesIn2ndLevel( zLayer, xLayer , colorMap);

        while ( iterations-- > 0 )
        {
            for ( int ii = 0 ; ii < xLayer.length ; ii++ )
            {
                for ( int jj = 0 ; jj < xLayer[ ii ].length ; jj++ )
                {
                    int zRowIdx = ii / numK, zColIdx = jj/numK;

                    List<Integer> neighbors = getNeighbors( xLayer, ii, jj );
                    int finalColorValue = decideColorGreyScaleEx( yLayer[ ii ][ jj ],
                            zLayer[ zRowIdx][ zColIdx ], colorMap, neighbors );
                    xLayer[ ii ][ jj ] = finalColorValue;
                }
            }

            for ( int ii = 0 ; ii < zLayer.length ; ii++ )
            {
                for ( int jj = 0 ; jj < zLayer[ ii ].length ; jj++ )
                {
                    List<Integer> neighborsOfZ = getNeighborsOfZi( xLayer, ii, jj );
                    int finalColorValue = decideColorForZPixelGS( zLayer[ ii ][ jj ], neighborsOfZ, colorMap );
                    zLayer[ii][jj] = finalColorValue;
                }
            }
        }
        return xLayer;
    }

    private int[][] denoisifyBWImageWith2HiddenLevels ( int[][] yLayer )
    {
        int[][] xLayer = new int[ yLayer.length ][ yLayer[ 0 ].length ];

        int row = ( yLayer.length / numK == yLayer.length / numK + yLayer.length % numK ) ? yLayer.length / numK :
                yLayer.length / numK + 1;
        int col = ( yLayer[ 0 ].length / numK == yLayer[ 0 ].length / numK + yLayer[ 0 ].length % numK ) ? yLayer[ 0
                ].length / numK : yLayer[ 0 ].length / numK + 1;

        double[][] zLayer = new double[ row ][ col ];
        HashMap<Integer, Integer> colorMap = ImageUtils.createColorMap( yLayer );

        initializeHiddenNodes( xLayer, yLayer );
        initializeHiddenNodesIn2ndLevel( zLayer, xLayer, colorMap );

        while ( iterations-- > 0 )
        {
            for ( int ii = 0 ; ii < xLayer.length ; ii++ )
            {
                for ( int jj = 0 ; jj < xLayer[ ii ].length ; jj++ )
                {
                    int zRowIdx = ii / numK, zColIdx = jj/numK;

                    List<Integer> neighbors = getNeighbors( xLayer, ii, jj );
                    int finalColorValue = decideColorBorWEx( yLayer[ ii ][ jj ],
                            zLayer[ zRowIdx][ zColIdx ], colorMap, neighbors );
                    xLayer[ ii ][ jj ] = finalColorValue;

                }
            }

            for ( int ii = 0 ; ii < zLayer.length ; ii++ )
            {
                for ( int jj = 0 ; jj < zLayer[ ii ].length ; jj++ )
                {
                    List<Integer> neighborsOfZ = getNeighborsOfZi( xLayer, ii, jj );
                    int finalColorValue = decideColorForZPixelBW( zLayer[ ii ][ jj ], neighborsOfZ, colorMap );
                    zLayer[ii][jj] = finalColorValue;
                }
            }
        }
        return xLayer;
    }

    private int[][] denoisifyGreyScaleImage ( int[][] observedImage )
    {
        int[][] hiddenNodes = new int[ observedImage.length ][ observedImage[ 0 ].length ];
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
        int[][] hiddenNodes = new int[ observedNodes.length ][ observedNodes[ 0 ].length ];
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
            correlations.put( correlation, color );
        }

        return correlations.get( correlations.lastKey() );
    }


    private int decideColorForZPixelBW (
            double observedNode,
            List<Integer> neighbors,
            HashMap<Integer, Integer> colorMap )
    {
        SortedMap<Double, Integer> energies = new TreeMap<Double, Integer>();

        for ( int color : colorMap.keySet() )
        {
            double energy = 0;
            for ( int neighbor : neighbors )
            {
                if ( neighbor == observedNode )
                {
                    energy += -omega;
                }
                else
                {
                    energy += omega;
                }
            }
            energies.put( energy, color );
        }

        return ( int ) ( energies.get( energies.firstKey() ) >= colorsAvg ? maxColor : minColor );
    }


    private int decideColorForZPixelGS (
            double observedNode,
            List<Integer> neighbors,
            HashMap<Integer, Integer> colorMap )
    {
        SortedMap<Double, Integer> energies = new TreeMap<Double, Integer>();

        for ( int color : colorMap.keySet() )
        {
            double energy = 0;
            for ( int neighbor : neighbors )
            {
                energy += (Math.log(1 + Math.abs(neighbor - observedNode)) - 1) * omega;
            }
            energies.put( energy, color );
        }

        return energies.get( energies.firstKey() );
    }


    private int decideColorBorWEx (
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
            correlations.put( correlation, color );
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
            correlations.put( correlation, color );
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
            double correlation = computeJointProbabilityForGreyScalePixelEx( observedNode, colorMap.get( color ),
                    hiddenLayer2Node,
                    neighbors );

            correlations.put( correlation, color );
        }

        return correlations.get( correlations.lastKey() );
    }


    private double computeJointProbabilityForGreyScalePixel (
            int observedNode,
            int hiddenNodeValue,
            List<Integer> hiddenNeighbors )
    {
        double potentialXiXj = 0, potentialXiYi;

        for ( Integer neighbor : hiddenNeighbors )
        {
            potentialXiXj += ( Math.log( Math.abs( hiddenNodeValue - neighbor ) + 1 ) - 1 ) *
                    beta;
        }
        potentialXiYi = ( ( Math.log( Math.abs( hiddenNodeValue - observedNode ) ) + 1 ) - 1 ) * eta;

        double energy = potentialXiXj + potentialXiYi;
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
        double energy = potentialXiXj + potentialXiYi + potentialXiZj;
        return Math.exp( -energy );
    }


    private double computeJointProbabilityForBAndWPixel (
            int observedNode,
            int hiddenNodeValue,
            List<Integer> hiddenNeighbors )
    {
        double beta_summationXiXj = 0, eta_XiYi;
        for ( Integer neighbor : hiddenNeighbors )
        {
            double betaLocal;
            betaLocal = neighbor == hiddenNodeValue ? -beta : beta;
            beta_summationXiXj += betaLocal;
        }

        eta_XiYi = hiddenNodeValue == observedNode ? -eta : eta;
        double energy = beta_summationXiXj + eta_XiYi;
        return Math.exp( -energy );
    }


    private double computeJointProbabilityForBAndWPixelEx (
            int observedNode,
            int hiddenNodeValue,
            double hiddenLayer2NodeValue,
            List<Integer> hiddenNeighbors )
    {
        double beta_summationXiXj = 0, eta_XiYi, omega_XiZj;
        for ( Integer neighbor : hiddenNeighbors )
        {
            double betaLocal;
            betaLocal = neighbor == hiddenNodeValue ? -beta : beta;
            beta_summationXiXj += betaLocal;
        }

        eta_XiYi = hiddenNodeValue == observedNode ? -eta : eta;
        omega_XiZj = hiddenNodeValue == hiddenLayer2NodeValue ? -omega : omega;
        double energy = beta_summationXiXj + eta_XiYi + omega_XiZj;

        return Math.exp( -energy );
    }


    private List<Integer> getNeighbors ( int[][] hiddenNodes, int rowIdx, int colIdx )
    {
        List<Integer> neighbors = new ArrayList<Integer>();
        if ( colIdx != 0 )
        {
            neighbors.add( hiddenNodes[ rowIdx ][ colIdx - 1 ] );  // left neighbor
        }
        if ( colIdx != hiddenNodes[ 0 ].length - 1 )
        {
            neighbors.add( hiddenNodes[ rowIdx ][ colIdx + 1 ] );  // right neighbor
        }

        if ( rowIdx != 0 )
        {
            neighbors.add( hiddenNodes[ rowIdx - 1 ][ colIdx ] );  // top neighbor
        }

        if ( rowIdx != hiddenNodes.length - 1 )
        {
            neighbors.add( hiddenNodes[ rowIdx + 1 ][ colIdx ] );  // bottom neighbor
        }
        return neighbors;
    }



    private List<Integer> getNeighborsOfZi ( int[][] hiddenLayerX, int rowIdx, int colIdx )
    {
        List<Integer> neighbors = new ArrayList<Integer>();
        /*for ( int ii = 0 ; ii < hiddenLayerX.length ; ii++ )
        {
            for ( int jj = 0 ; jj < hiddenLayerX[ ii ].length ; jj++ )
            {
                if(isConnected(ii, jj, rowIdx, colIdx, numK))
                    neighbors.add(hiddenLayerX[ii][jj]);
            }
        }*/

        int rowLimit = Math.min( rowIdx * numK + numK, hiddenLayerX.length / numK + numK );
        int colLimit = Math.min( colIdx * numK + numK, hiddenLayerX[ 0 ].length / numK + numK );

        for ( int i = rowIdx * numK ; i < rowLimit ; i++ )
        {
            for ( int j = colIdx * numK ; j < colLimit ; j++ )
            {
                neighbors.add( hiddenLayerX[ i ][ j ] );
            }
        }
        return neighbors;
    }


    private void initializeHiddenNodes ( int[][] hiddenNodes, int[][] observedNodes )
    {
        for ( int ii = 0 ; ii < observedNodes.length ; ii++ )
        {
            System.arraycopy( observedNodes[ ii ], 0, hiddenNodes[ ii ], 0, observedNodes[ ii ].length );
        }
    }


    private double eta;
    private double beta;
    private double omega;
    private double maxColor;
    private double minColor;
    private double colorsAvg;

    private int numK;
    private int iterations;

    private boolean useSecondLevel;
}

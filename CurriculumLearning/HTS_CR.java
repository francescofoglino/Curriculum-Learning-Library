package CurriculumLearning;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import CurriculumLearning.CurriculumLearning_UTILS.CurriculumGenerator;
import GridWorld.GridWorldTransferLearning;

/** Class for the algorithm HTS_CR */

public class HTS_CR{

  CurriculumGenerator                   currGen;
  HashMap<ArrayList<Integer>, Double>   evaluatedCurricula;
  int                                   currMaxLen;
  int                                   numEph;
  ArrayList<Integer>					sourcesIdxs;
  double 								stopPerformance;
  long  								stopIterations;
  long 									stopActions;
  double 								tttThreshold;
  
  double 								bestPerformance;
  ArrayList<Integer>					bestCurriculum;
  long 									totalNumActions;
  
  /** Simple constructor */
  HTS_CR(CurriculumGenerator gen, int maxLen, int nEpochs, double maxPerf, long maxIter, double threshold, long maxNumActions){
    currGen             = gen;
    evaluatedCurricula  = new HashMap<ArrayList<Integer>, Double>();
    currMaxLen          = maxLen;
    stopActions 		= maxNumActions;
    numEph              = nEpochs;
    
    sourcesIdxs = new ArrayList<Integer>();
    
    for(int i=0; i<currGen.sourceSet.size(); ++i)
    	sourcesIdxs.add(i);
    
    stopPerformance 	= maxPerf;
    stopIterations		= maxIter;
    tttThreshold 		= threshold;
  }
  
  /** Most important function which actually runs the HTS-CR algorithm */
  public HashMap<String, Object> findBestCurriculum(){

    int numSources = currGen.sourceSet.size(), currMaxLen = 3;
    List<Integer> heads, tails;
    
    /** Obtain performance of all the curricula formed by the combination of two sources */
    HashMap<ArrayList<Integer>, Double> pairsRanking = evaluatePairs();
    
    if(stoppingCriterionMet())
    	return returnData();

    List<Integer> sourcesAsHeads = bestHeads(pairsRanking);
    List<Integer> sourcesAsTails = bestTails(pairsRanking);

    int headsNumber = 1, tailsNumber = 1;

    for(int hyper = 1; hyper <= 2*(numSources-1); ++hyper){
      /** For each iteration we increase either the number of heads or the number of tails considered */
      if(hyper%2 == 1)
        ++headsNumber;
      else
        ++tailsNumber;

      /** Note: the second index specified in subList is not included in the resultant list */
      heads = sourcesAsHeads.subList(0, headsNumber);
      tails = sourcesAsTails.subList(0, tailsNumber);

      for(int l = 3; l <= currMaxLen; ++l){

        for(int h = 0; h < headsNumber; ++h){

          int hd = sourcesAsHeads.get(h);

          for(int t = 0; t < tailsNumber; ++t){

            int tl = sourcesAsTails.get(t);

            if(hd != tl){

              List<Integer> leftOvers = uniqueElements(hd, heads, tl, tails);

              ArrayList<ArrayList<Integer>> leftOversPermutations = permutations(leftOvers, l-2);
              //if(leftOvers.size() != 0 && leftOvers.size() != 1){}
              
              /** Build the actual curriculum to evaluate following the structure (head) -> (leftOversPermutations[i]) -> (tail) */
              for(List<Integer> p: leftOversPermutations){
                ArrayList<Integer> curr = new ArrayList<Integer>(p);
                curr.add(0, hd);
                curr.add(tl);

                if(!evaluatedCurricula.containsKey(curr))
                  evaluatedCurricula.put(curr, evaluate(curr));
                
                if(stoppingCriterionMet())
                	return returnData();
              }

            }

          }
        }
      }

    }
    
    return null;
  }
  
  private boolean stoppingCriterionMet(){
	  
	  if(bestPerformance >= stopPerformance || evaluatedCurricula.size() >= stopIterations || totalNumActions >= stopActions)
		  return true;
	  
	  return false;
  }
  
  /** Calculate all the permutations of the given list of elements in k positions */
  public ArrayList<ArrayList<Integer>> permutations(List<Integer> elements, int k){

    ArrayList<ArrayList<Integer>> allPerms = new ArrayList<ArrayList<Integer>>(), perms = new ArrayList<ArrayList<Integer>>();

    if(k == 1){
      for(int e : elements)
        allPerms.add(new ArrayList<Integer>() {{add(e);}});
    }else{

      for(Integer e : elements){
    	
    	ArrayList<Integer> subArray = new ArrayList(elements);
    	subArray.remove(e);
    	  
        perms = permutations(subArray, k-1);

        for(ArrayList<Integer>pp: perms){
        	pp.add(0,e);
        	allPerms.add(pp);
        }
      }
    }

    return allPerms;
  }

  public HashMap<ArrayList<Integer>, Double> evaluatePairs(){

    HashMap<ArrayList<Integer>, Double> evaPairs = new HashMap<ArrayList<Integer>, Double>();

    ArrayList<ArrayList<Integer>> pairs = permutations(sourcesIdxs, 2);

    for(ArrayList<Integer> p: pairs){

      double eva = evaluate(p);
      
      /** This statement is to make sure that the first evaluated pair becomes the best one
       *  since there are no initializations for bestPerformance or bestCurriculum which could
       *  result in a wrong true value from the function stoppingCriterionMet() */
      if(evaluatedCurricula.size() == 0) {
    	  bestPerformance 	= eva;
    	  bestCurriculum 	= p;
      }
      
      evaPairs.put(p, eva);

      if(!evaluatedCurricula.containsKey(p))
        evaluatedCurricula.put(p, eva);
    }

    return evaPairs;
  }

  public List<Integer> uniqueElements(Integer h, List<Integer> heads, Integer t, List<Integer> tails){
    List<Integer> hs = new ArrayList<Integer>(heads);
    List<Integer> ts = new ArrayList<Integer>(tails);
    
	hs.remove(h);
    hs.remove(t);
    ts.remove(h);
    ts.remove(t);

    ArrayList<Integer> unique = new ArrayList<Integer>(hs);
    unique.removeAll(ts);
    unique.addAll(ts);

    return unique;
  }

  private double evaluate(ArrayList<Integer> curriculum){

    /** Re-initialize the performance analyser for each evaluation */
    currGen.currPA = (new GridWorldTransferLearning()).new PerformanceAnalyser();
    
    currGen.performCurriculumFromArray(numEph, curriculum);
    
    currGen.currPA.init();
    
    double performance = currGen.currPA.cumulativeReturnStats().get("Mean");
    
    if(performance > bestPerformance) {
    	bestPerformance = performance;
    	bestCurriculum	= curriculum;
    }
    
    double numSamples = currGen.currPA.strongTimeToThresholdStats(tttThreshold).get("Mean");
    totalNumActions  += numSamples;
    
    return performance;
  }
  
  private ArrayList<Object> gradePairs(HashMap<Object, Number> pairsRanking, boolean asHeads){
	  
	  int hot = 0;
	  
	  if(!asHeads)
		  hot = 1;
	  
	  /** Sort all the 2-sources curricula from the best to the worst*/
	  ArrayList<Object> sortedPairs = bubbleSortMapKeys(pairsRanking);
	  
	  /** Rank all sources by how well they perform as head or tail in a curriculum */
	  HashMap<Object, Number> sourcesRankingHoT = new HashMap<Object, Number>();
	  
	  for(int j=0; j<sortedPairs.size(); ++j){
		  
		  ArrayList<Integer> pair 	= ((ArrayList<Integer>)sortedPairs.get(j));
		  Integer sourceHoT 		 	= pair.get(hot);
		  
		  /** If the element is not already in the map then we initialize it */
		  int currentScore = 0;
		  if(sourcesRankingHoT.get(sourceHoT) != null)
			  currentScore = (Integer)sourcesRankingHoT.get(sourceHoT);
		  else
			  sourcesRankingHoT.put(sourceHoT, 0);
		  
		  sourcesRankingHoT.replace(((ArrayList<Integer>)sortedPairs.get(j)).get(hot), currentScore - j);
	  }
	  
	  /** Sort all sources depending on their performance as head/tail */
	  return bubbleSortMapKeys(sourcesRankingHoT);
	  
  }
  
  /** Sorts the map keys in ascendent order depending on their value */
  private ArrayList<Object> bubbleSortMapKeys(HashMap<Object, Number> map){
	  
	  ArrayList<Object> sortedKeys = new ArrayList<Object>(); 
	  
	  for(Object key:map.keySet()){
		  if(sortedKeys.isEmpty())
			  sortedKeys.add(key);
		  else{
			  
			  Number value = map.get(key);
			  
			  for(int i=0; i<sortedKeys.size(); ++i){
				  if(value.doubleValue() > map.get(sortedKeys.get(i)).doubleValue()){
					  sortedKeys.add(i, key);
					  break;
				  }else {
					  if(i == sortedKeys.size()-1) {
						  sortedKeys.add(key);
						  break;
					  }
				  }  
			  }
		  }
	  }
	  
	  return sortedKeys;
	  
  }
  
  /** Two identically opposite algorithms for extracting the sorted list of best heads and tails */
  public ArrayList<Integer> bestHeads(HashMap<ArrayList<Integer>, Double> pairsRanking){
	  
	  HashMap<Object, Number> pairsRankingON = new HashMap<Object, Number>(pairsRanking);
	  
	  ArrayList<Integer> heads = new ArrayList<Integer>();
	  
	  ArrayList<Object> gradedPairs = gradePairs(pairsRankingON, true);
	  
	  for(Object s:gradedPairs)
		  heads.add((Integer)s);
	  
	  return heads;
  }
  
  public ArrayList<Integer> bestTails(HashMap<ArrayList<Integer>, Double> pairsRanking){
	  
	  HashMap<Object, Number> pairsRankingON = new HashMap<Object, Number>(pairsRanking);
	  
	  ArrayList<Integer> tails = new ArrayList<Integer>();
	  
	  ArrayList<Object> gradedPairs = gradePairs(pairsRankingON, false);
	  
	  for(Object s:gradedPairs)
		  tails.add((Integer)s);
	  
	  return tails;
  }
  /** ------------------------------------------------------------------------------------------ */
  
  private HashMap<String, Object> returnData() {
	  
	  	HashMap<String, Object> stats = new HashMap<String, Object>();
		stats.put("Curriculum", bestCurriculum);
		stats.put("Performance", bestPerformance);
		stats.put("Iterations", evaluatedCurricula.size());
		stats.put("Actions", totalNumActions);
		
		return stats;
  }
  
}

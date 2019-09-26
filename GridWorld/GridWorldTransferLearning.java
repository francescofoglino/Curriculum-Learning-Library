package GridWorld;

import TransferLearning.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

import TransferLearning.TransferLearning;
import burlap.behavior.functionapproximation.sparse.tilecoding.TileCodingFeatures;
import burlap.behavior.learningrate.ConstantLR;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam;
import burlap.behavior.valuefunction.QProvider;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldDomain.AtLocationPF;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.oo.state.OOState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
import burlap.domain.singleagent.gridworld.state.GridLocation;

public class GridWorldTransferLearning implements TransferLearning{
	
	/** GridWorldTransferLearning variables */
	GridWorldDomain gw;
	OOSADomain domain;
	RewardFunction rf;
	TerminalFunction tf;
	StateConditionTest goalCondition;
	State initialState;
	HashableStateFactory hashingFactory;
	SimulatedEnvironment env;
	String refTag;
	int nTilings;
	public PerformanceAnalyser pa;
	/*****************************************/
	
	public GridWorldTransferLearning() {}
	
	public GridWorldTransferLearning(GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf, int numTilings) {
		
		refTag = gwdf.tag;
 		
		GridWorldState state0 = gwdf.generateInitialState();
		
		tf = new GridWorldTerminalFunction(((Integer)state0.get("treasure:x")), ((Integer)state0.get("treasure:y")));
		
		for(int p = 1; p <= gwdf.pits.size(); ++p)//set all the pits to terminal states
			((GridWorldTerminalFunction)tf).markAsTerminalPosition(((Integer)state0.get("pit" + p + ":x")), ((Integer)state0.get("pit" + p + ":y")));
		
		gw = new GridWorldDomain(gwdf.map);
		gw.setTf(tf);
		
		gw.setRf(new GridWorldRF(gw, gwdf));
		
		goalCondition = new TFGoalCondition(tf);
		
		domain = gw.generateDomain();
		
		initialState = state0;
		
		/** instantiate the HashableStateFactory */
		hashingFactory = new SimpleHashableStateFactory();
		
		/** creation of the actual environment with which our agent will interact */
		env = new SimulatedEnvironment(domain, initialState);
		
		nTilings = numTilings;
	}
	
	/** Main learning function where all the most important features and parameters are set on our agent so that we can run learning episodes and epochs */
	public TLLinearVFA PerformLearningAlgorithm(String outputPath, int nEpochs, int nEpisodes, TLLinearVFA...sourceVFAs) {

		File outputFile = new File(outputPath);
		outputFile.mkdirs();
		
		TLLinearVFA vfa = null;
		
		State stateToPrint = initialState;
		
		/** Declaration of output files of diverse purposes */
		PrintWriter osAV = null;
		try {
			osAV = new PrintWriter(outputPath+"/ActionValueFunction.txt");
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		
		PrintWriter osR = null;
		try {
			osR = new PrintWriter(outputPath+"/Return.txt");
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		/** *********************************************** */
		
		this.pa = new PerformanceAnalyser();
		
		for(int epoch=0; epoch<nEpochs; ++epoch){
			
			/** Parameters for learning algorithm/agent */
			double gamma = 				0.999;
			double alpha = 				0.1;
			double alphaTilingWise = 	0;		/** This is the alpha you associate with the tilings in order to "distribute" the general alpha */
			double lambda = 			0.9;
			double qInit = 				0.; 	/** Q-values initial value
			
			/** For collecting data during epochs */
			EpochsData collector = new EpochsData();
			
			/** to change */
			GradientDescentSarsaLam agent = null;
			String writeString = "";
			
			/** Here we initialize the function approximator (Tile Coding), perform value-function transfer if any previous value-function
			 *  is present, and finally we select the learning algorithm (Sarsa(lambda)). This whole part has to be modified/adapted in case
			 *  you want  to use a different learning algorithm */
				
			TLTileCodingFeatures tileCoding = GridWorldTransferLearning_UTILS.CreateTransferLearningTCFeatures(refTag, nTilings);
			vfa = new TLLinearVFA(tileCoding);
			
			/** Transfer Learning */
			if(sourceVFAs != null){
				/** These two lines fix the problem with the input sourceVFAs when passed as a variable X = null */
				if(sourceVFAs.length == 1 && sourceVFAs[0] == null)
					sourceVFAs = null;
				else{
				lambda = 0.9;
				
				/** Loops over all the source value-functions for transferring knowledge in parallel in case you pass more than one 
				 * (so you want to learn from two or more sources together) */
				for(TLLinearVFA sVFA : sourceVFAs)
					sVFA.transferVFA(vfa, null);
				}
			}
			
			alphaTilingWise = alpha/((TLTileCodingFeatures)vfa.sparseStateFeatures).numTIlings();
			
			agent = new GradientDescentSarsaLam(domain, gamma, vfa, alphaTilingWise, lambda);
			
			writeString = "TL_";
			
			stateToPrint = initialState;
			
			/*********************************************************************************************************************************/
			
			QProvider qp = (QProvider)agent;
			
			/** Starting value for epsilon. This will be lowered to 0 towards the end of the epoch */
			double epsilon = 0.1;
			EpsilonGreedy lEpsGrdy = new EpsilonGreedy(agent, epsilon);
			agent.setLearningPolicy(lEpsGrdy);
			
			/** Looping on the episodes */
			for(long episode = 0; episode < nEpisodes; episode++){
				
				if(outputPath != null)
					GridWorldTransferLearning_UTILS.WriteActionValues(osAV, qp, stateToPrint, episode);
				
				/***************************************** Shaping epsilon here *****************************************/
				
				double explorationRatio = 15./20;		/** fraction of episodes we run with initial exploration value */
				double lowerExplorationRatio = 4./20;	/** fraction of episodes we run while lowering epsilon */
				
				double stopExp = (explorationRatio+lowerExplorationRatio)*nEpisodes;	/** the episode when we stop exploration */
				double startExp = nEpisodes*explorationRatio; 							/** when the lowering starts to be applied */
				double intervalEp = stopExp - startExp;  								/** interval of episodes where we decrease the value of epsilon */
				
				if(episode>=(startExp) && episode <= stopExp){
					epsilon = epsilon * (stopExp - episode)/intervalEp;
					
					if(agent instanceof GradientDescentSarsaLam){
						GradientDescentSarsaLam agentQLearning = (GradientDescentSarsaLam)agent;
						
						EpsilonGreedy newLEpsGrdy = new EpsilonGreedy(agent, epsilon);
						agent.setLearningPolicy(newLEpsGrdy);
					}else{
						System.out.println("ERROR! You are trying to change the learning policy to an invalid agent!");
					}
				}else if(episode > stopExp){
					epsilon = 0.;
					
					GradientDescentSarsaLam agentQLearning = (GradientDescentSarsaLam)agent;
					
					EpsilonGreedy newLEpsGrdy = new EpsilonGreedy(agent, epsilon);
					agent.setLearningPolicy(newLEpsGrdy);
				}
				
				/********************************************************************************************************/	
				
				/** the second input is the number of actions per episode */
				Episode e = agent.runLearningEpisode(env,50);
				
				/** print episode data (for replay with visualizer )*/
				e.write(outputPath + writeString + "_" + epoch + "_" + episode);		
				
				double r = e.discountedReturn(1); /** not discounting the Return in this phase as this is just for printing purposes */
				double a = agent.getLastNumSteps();
				
				collector.returns.add(r);
				collector.numberOfActions.add(a);
				
				if(outputPath != null)
					osR.println(episode + " " + r + " " + a);
				
				/** reset environment for next learning episode */
				env.resetEnvironment();
				
				if(episode % 10 == 0 || episode == nEpisodes-1){
					
					((TLTileCodingFeatures)vfa.sparseStateFeatures).setLearning(false);
					
					double score = GridWorldTransferLearning_UTILS.PerformLearntPolicy(agent, env, outputPath, "policyEvaluation.txt", episode);
					collector.policyScores.add(score);
					
					agent.setLearningRate( new ConstantLR(alphaTilingWise));
					agent.setLearningPolicy(lEpsGrdy);
					
					((TLTileCodingFeatures)vfa.sparseStateFeatures).setLearning(true);
				}
				
			}
			
			pa.epochs.add(collector);
		}
		
		pa.init();
		
		osAV.close();
		osR.close();
		
		return vfa;
	}
	
	/** tool for visualizing the result of the learning */
	public void visualize(String outputPath){
		Visualizer v = GridWorldVisualizer.getVisualizer(gw.getMap());
		new EpisodeSequenceVisualizer(v, domain, outputPath);
	}

	/** Design of the Reward Function for this domain. This implementation specifically assigns the following rewards:
	 *  - 	+ 200 	for reaching the treasure (terminal state)
	 *  - 	- 2500 	for falling into a pit (terminal state)
	 *  -	- 250 	for stepping next to a fire
	 *  - 	- 500	for stepping on a fire
	 *  - 	- 1 	all the other cases */
	public class GridWorldRF implements RewardFunction{
		
		GridWorldDomain domain;
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf;
		
		GridWorldRF(GridWorldDomain domain, GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf){
			this.domain = domain;
			this.gwdf = gwdf;
		};
		
		@Override
		public double reward(State s, Action a, State sprime) {
			AtLocationPF al = domain.new AtLocationPF(GridWorldDomain.CLASS_LOCATION, new String[]{GridWorldDomain.CLASS_AGENT});
			
			/** if the current action leads us to the goal */
			if(al.isTrue(((OOState)sprime), new String[]{GridWorldDomain.CLASS_AGENT,"treasure"}))
				return 200;
			else {
				
				/** if the current action leads us to a pit */
				for(int p = 1; p <= gwdf.pits.size(); ++p) {
					if(al.isTrue(((OOState)sprime), new String[]{GridWorldDomain.CLASS_AGENT,"pit" + p}))
						return -2500;
				}
				

				boolean onFire = false, nextToFire = false;
				
				for(int f = 1; f <= gwdf.fires.size(); ++f) {					
					if(al.isTrue(((OOState)sprime), new String[]{GridWorldDomain.CLASS_AGENT,"fire" + f}))
						onFire = true;
					/** from inside a fire whichever action I take leads me in a "next to fire" state */
					else if(al.isTrue(((OOState)s), new String[]{GridWorldDomain.CLASS_AGENT,"fire" + f}) || manhattanDistance((GridWorldState)sprime, f) == 1)
						nextToFire = true; 
				}
				
				if(onFire)
					return -500;
				else if(nextToFire)
					return -250;
				
				return -1;
			}
			
		}
		
		private int manhattanDistance(GridWorldState state, int i) {
			
			int dx = Math.abs(((Integer)state.get("fire" + i + ":x")) - ((GridWorldState)state).agent.x);
			int dy = Math.abs(((Integer)state.get("fire" + i + ":y")) - ((GridWorldState)state).agent.y);
			
			return dx + dy;
		}
	}
	
	/** EpochsData is a class for collecting all the data about 1 learning epoch. It also provides basic performance functions for elaborating the collected data */
	public class EpochsData{
		
		public ArrayList<Double> returns;
		public ArrayList<Double> numberOfActions;
		public ArrayList<Double> policyScores; /** the returns accumulated by the agent during the policy evaluation steps */
		
		EpochsData() {
			returns 		= new ArrayList<Double>();
			numberOfActions = new ArrayList<Double>();
			policyScores	= new ArrayList<Double>();
		}
		
		EpochsData(ArrayList<Double> Rs, ArrayList<Double> As, ArrayList<Double> Es) {
			returns 		= Rs;
			numberOfActions = As;
			policyScores	= Es;
		}
		
		public double getCumulativeActions() {
			
			double allActions = 0;
			for(double a : numberOfActions) {
				allActions += a; 
			}
			
			return allActions;
		}
		
		/** Below there are the 4 functions for calculating the 4 performance */
		public double cumulativeReturn() {
			double cumRet = 0;
			for(double R : returns)
				cumRet += R;
			
			return cumRet;
		}
		
		public double jumpStart() {
			return returns.get(0);
		}
		
		public double maxReturn() {
			double maxR = policyScores.get(0);			
			for(double R : policyScores) {
				if( R > maxR)
					maxR = R;
			}
			
			return maxR;
		}
		
		public double weakTimeToThreshold(double threshold) {
			
			double cumulativeActions = 0;
			
			for(int i = 0; i < policyScores.size(); ++i) {
				if(policyScores.get(i) >= threshold) {
					
					/** Since we run a policy evaluation step each 10 learning steps apart from the last one, we need to take it into
					 *  account here. Example: learning steps -> (0 ,1, ... 98, 99) evaluation steps -> (0, 10, 20, ... 80, 90, 99) */
					if(i*10 >= numberOfActions.size()) {
						
						for (int j = 0; j < numberOfActions.size(); ++j)
							cumulativeActions += numberOfActions.get(j);
						
						return cumulativeActions;
					}else {
						
						for (int j = 0; j < i*10; ++j)
							cumulativeActions += numberOfActions.get(j);
						
						return cumulativeActions;
					}	
				}
					
			}
			return -1; /** dummy value in case the agent does not hit the threshold during the given time */
		}
	}
	
	/** PerformanceAnalyser is a class for interacting with multiple EpochsData in order to extract statistically relevant data */
	public class PerformanceAnalyser{
		
		public ArrayList<EpochsData> 	epochs;
		private int 					n; 		/** number of samples */
		private double 					t; 		/** t value from t distribution */
		public ArrayList<Double> 		sourceActions;
		
		public PerformanceAnalyser(){
			epochs 			= new ArrayList<EpochsData>();
			sourceActions	= new ArrayList<Double>();
		}
		
		public void init() {
			n = epochs.size();
			
			/** We consider a number of epochs equal to 1, 3, 5, 10 or 20. Any other value is not handled for simplicity */
			switch (n-1) {
				case 0:
					t = 0;
					break;
				
				case 2:
					t = 4.303;
					break;
				
				case 4:
					t = 2.776;
					break;
				
				case 9:
					t = 2.262;
					break;
					
				case 19:
					t = 2.093;
					break;
					
				default:
					System.err.println("The number of epochs must be either 1, 3, 5, 10 or 20. \n\nTerminating...");
					System.exit(0);
			}
		}
		
		/** Calculate Mean, Variance and  Confidence Interval for the Cumulative Return performance for the given epochs*/
		public HashMap<String, Double> cumulativeReturnStats() {
							
			/************** Calculate the Mean Value **************/
			double mean = 0;
			
			for(int e = 0; e < n; ++e)
				mean += epochs.get(e).cumulativeReturn();
			
			mean = mean/n;
			/******************************************************/
			
			/*************** Calculate the Variance ***************/
			double numerator = 0;
			
			for(int e = 0; e < n; ++e)
				numerator += Math.pow(epochs.get(e).cumulativeReturn() - mean, 2);
			
			double variance = numerator/(n-1);
			/******************************************************/
			
			/********** Calculate the Confidence Interval *********/
			double ci = t*(Math.sqrt(variance)/Math.sqrt(n));
			/******************************************************/
			
			HashMap<String, Double> stats = new HashMap<String, Double>();
			stats.put("Mean", mean);
			stats.put("Variance", variance);
			stats.put("CI", ci);
			
			return stats;
		}
		
		/** Calculate Mean, Variance and  Confidence Interval for the Jump Start performance for the given epochs*/
		public HashMap<String, Double> jumpStartStats() {
			
			/************** Calculate the Mean Value **************/
			double mean = 0;
			
			for(int e = 0; e < n; ++e)
				mean += epochs.get(e).jumpStart();
			
			mean = mean/n;
			/******************************************************/
			
			/*************** Calculate the Variance ***************/
			double numerator = 0;
			
			for(int e = 0; e < n; ++e)
				numerator += Math.pow(epochs.get(e).jumpStart() - mean, 2);
			
			double variance = numerator/(n-1);
			/******************************************************/
			
			/********** Calculate the Confidence Interval *********/
			double ci = t*(Math.sqrt(variance)/Math.sqrt(n));
			/******************************************************/
			
			HashMap<String, Double> stats = new HashMap<String, Double>();
			stats.put("Mean", mean);
			stats.put("Variance", variance);
			stats.put("CI", ci);
			
			return stats;
		}
		
		/** Calculate Mean, Variance and  Confidence Interval for the Max Return performance for the given epochs*/
		public HashMap<String, Double> maxReturnStats() {
			
			/************** Calculate the Mean Value **************/
			double mean = 0;
			
			for(int e = 0; e < n; ++e)
				mean += epochs.get(e).maxReturn();
			
			mean = mean/n;
			/******************************************************/
			
			/*************** Calculate the Variance ***************/
			double numerator = 0;
			
			for(int e = 0; e < n; ++e)
				numerator += Math.pow(epochs.get(e).maxReturn() - mean, 2);
			
			double variance = numerator/(n-1);
			/******************************************************/
			
			/********** Calculate the Confidence Interval *********/
			double ci = t*(Math.sqrt(variance)/Math.sqrt(n));
			/******************************************************/
			
			HashMap<String, Double> stats = new HashMap<String, Double>();
			stats.put("Mean", mean);
			stats.put("Variance", variance);
			stats.put("CI", ci);
			
			return stats;
		}
		
		/** Calculate Mean, Variance and  Confidence Interval for the Time To  performance for the given epochs*/
		public HashMap<String, Double> weakTimeToThresholdStats(double threshold) {
			
			/************** Calculate the Mean Value **************/
			double mean = 0;
			double wttt;
			
			int fails = 0;
			
			for(int e = 0; e < n; ++e) {
				
				wttt = epochs.get(e).weakTimeToThreshold(threshold);
				
				if(wttt == -1)
					++fails;
				
				mean += wttt;
			}
			
			if(fails != 0) {
				System.err.println("WARNING: the agent did not reach the threshold value in " + fails + " epochs.\n"
						+ "Impossible to compute the TTT statistical data for this experiment. Try changing the threshold value\n");
				
				return null;
			}
			
			mean = mean/n;
			/******************************************************/
			
			/*************** Calculate the Variance ***************/
			double numerator = 0;
			
			for(int e = 0; e < n; ++e)
				numerator += Math.pow(epochs.get(e).weakTimeToThreshold(threshold) - mean, 2);
			
			double variance = numerator/(n-1);
			/******************************************************/
			
			/********** Calculate the Confidence Interval *********/
			double ci = t*(Math.sqrt(variance)/Math.sqrt(n));
			/******************************************************/
			
			HashMap<String, Double> stats = new HashMap<String, Double>();
			stats.put("Mean", mean);
			stats.put("Variance", variance);
			stats.put("CI", ci);
			
			return stats;
		}
		
		/** Calculate Mean, Variance and  Confidence Interval for the Time To  performance for the given epochs*/
		public HashMap<String, Double> strongTimeToThresholdStats(double threshold) {
			
			/************** Calculate the Mean Value **************/
			double mean = 0;
			double wttt, sttt;
			
			int fails = 0;
			
			for(int e = 0; e < n; ++e) {
				
				wttt = epochs.get(e).weakTimeToThreshold(threshold);
				
				if(wttt == -1)
					++fails;
				
				sttt = wttt + sourceActions.get(e);
				
				mean += sttt;
			}
			
			if(fails != 0) {
				System.err.println("WARNING: the agent did not reach the threshold value in " + fails + " epochs.\n"
						+ "Impossible to compute the TTT statistical data for this experiment. Try changing the threshold value\n");
				
				return null;
			}
			
			mean = mean/n;
			/******************************************************/
			
			/*************** Calculate the Variance ***************/
			double numerator = 0;
			
			for(int e = 0; e < n; ++e)
				numerator += Math.pow(epochs.get(e).weakTimeToThreshold(threshold) - mean, 2);
			
			double variance = numerator/(n-1);
			/******************************************************/
			
			/********** Calculate the Confidence Interval *********/
			double ci = t*(Math.sqrt(variance)/Math.sqrt(n));
			/******************************************************/
			
			HashMap<String, Double> stats = new HashMap<String, Double>();
			stats.put("Mean", mean);
			stats.put("Variance", variance);
			stats.put("CI", ci);
			
			return stats;
		}
		
	}
	
	public static void main(String[] args) {
		
		System.out.println("Welcome to GridWorldTransferLearning!");
		
		/*******************************************************************************************************
		 *    _   _  ____     _____ _    _ _____  _____  _____ _____ _    _ _     _    _ __  __ 
		 *	 | \ | |/ __ \   / ____| |  | |  __ \|  __ \|_   _/ ____| |  | | |   | |  | |  \/  |
		 *	 |  \| | |  | | | |    | |  | | |__) | |__) | | || |    | |  | | |   | |  | | \  / |
		 * 	 | . ` | |  | | | |    | |  | |  _  /|  _  /  | || |    | |  | | |   | |  | | |\/| |
		 *	 | |\  | |__| | | |____| |__| | | \ \| | \ \ _| || |____| |__| | |___| |__| | |  | |
		 *	 |_| \_|\____/   \_____|\____/|_|  \_\_|  \_\_____\_____|\____/|______\____/|_|  |_|
         *                                                                           
         * Simple Reinforcement Learning on one task. No curriculum or transfer involved
         *******************************************************************************************************/
		
		/** Choose here the Reinforcement Learning problem you want to solve and the number of tilings you want to 
		 *  use for implementing the tile coding function approximator for this specific problem. Nextly you need to 
		 *  specify the number of episodes and epochs*/
		GridWorldTransferLearning targetNoTransfer = new GridWorldTransferLearning(new GridWorldTransferLearning_UTILS().gwdf_test_M, 4);
		
		int nEpochs_targetNoTransfer = 20;
		int nEpisodes_targetNoTransfer = 100;
		
		TLLinearVFA targetVFANoTransfer =  targetNoTransfer.PerformLearningAlgorithm("temp/noTransfer/", nEpochs_targetNoTransfer, nEpisodes_targetNoTransfer, null);
		
		/** For visualizing all the samples of your experiment after learning */
		targetNoTransfer.visualize("temp/noTransfer/");
		
		/** For printing on terminal the performance of your agent */
		targetNoTransfer.pa.init();
		
		System.out.println("NO CURRIMCULUM RESULTS:\n\n");
		System.out.println("Weak TTT:\t\t" + targetNoTransfer.pa.weakTimeToThresholdStats(190));
		System.out.println("Max Return:\t\t" + targetNoTransfer.pa.maxReturnStats());
		System.out.println("Cumulative Return:\t" + targetNoTransfer.pa.cumulativeReturnStats());
		System.out.println("Jump Start:\t\t" + targetNoTransfer.pa.jumpStartStats());
		/*******************************************************************************************************/
		
		/*******************************************************************************************************
		 *	 _____ _    _ _____  _____  _____ _____ _    _ _     _    _ __  __ 
		 *  / ____| |  | |  __ \|  __ \|_   _/ ____| |  | | |   | |  | |  \/  |
		 * | |    | |  | | |__) | |__) | | || |    | |  | | |   | |  | | \  / |
		 * | |    | |  | |  _  /|  _  /  | || |    | |  | | |   | |  | | |\/| |
		 * | |____| |__| | | \ \| | \ \ _| || |____| |__| | |___| |__| | |  | |
		 *  \_____|\____/|_|  \_\_|  \_\_____\_____|\____/|______\____/|_|  |_|
		 *  
		 * Simple Curriculum Learning example. This usage of the code should be limited to test a new curriculum
         *******************************************************************************************************/                                                                     
		
		/** SOURCE 0 */
		GridWorldTransferLearning source = new GridWorldTransferLearning(new GridWorldTransferLearning_UTILS().gwdf_treasure, 4);
		
		int nEpochs_source = 1;
		int nEpisodes_source = 30;
		
		TLLinearVFA sourceVFA =  source.PerformLearningAlgorithm("temp/source/", nEpochs_source, nEpisodes_source, null);
		
		//source.visualize("temp/source/");
		
		/** SOURCE 1 */
		GridWorldTransferLearning source1 = new GridWorldTransferLearning(new GridWorldTransferLearning_UTILS().gwdf_fire, 4);
		
		int nEpochs_source1 = 1;
		int nEpisodes_source1 = 80;
		
		TLLinearVFA source1VFA =  source1.PerformLearningAlgorithm("temp/source1/", nEpochs_source1, nEpisodes_source1, sourceVFA);
		
		//source1.visualize("temp/source2/");
		
		/**TARGET */
		GridWorldTransferLearning target = new GridWorldTransferLearning(new GridWorldTransferLearning_UTILS().gwdf_test_M, 4);
		
		int nEpochs_target = 1;
		int nEpisodes_target = 500;
		
		TLLinearVFA targetVFA =  target.PerformLearningAlgorithm("temp/target/", nEpochs_target, nEpisodes_target, source1VFA);
		
		target.visualize("temp/target/");
		
		/** For printing on terminal the performance of your agent */
		target.pa.init();
		
		System.out.println("\n\n\nCURRIMCULUM RESULTS:\n\n");
		System.out.println("Weak TTT:\t\t" + target.pa.weakTimeToThresholdStats(150));
		System.out.println("Max Return:\t\t" + target.pa.maxReturnStats());
		System.out.println("Cumulative Return:\t" + target.pa.cumulativeReturnStats());
		System.out.println("Jump Start:\t\t" + target.pa.jumpStartStats());
	}

}

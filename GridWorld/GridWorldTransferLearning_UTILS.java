package GridWorld;

import TransferLearning.*;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

import TransferLearning.VariableDomainFloatingNorm;
import burlap.behavior.functionapproximation.sparse.tilecoding.TilingArrangement;
import burlap.behavior.learningrate.ConstantLR;
import burlap.behavior.policy.GreedyDeterministicQPolicy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.QValue;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.oo.state.OOState;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;

public class GridWorldTransferLearning_UTILS {
	
	/************************************** ONE-AXIS MAPS ******************************************/
	
	/** MAP_treasure_X
	 * 
	 * O O O O 
	 * O A T O
	 * O O O O
	 * */
	
	public static final int[][] MAP_treasure_X =   {{1,1,1},
													{1,0,1},
													{1,0,1},
													{1,1,1}};
	
	public static String tag_treasure_X = "treasure_X";
	
	public static ArrayList<GridLocation> allFires_treasure_X = new ArrayList<GridLocation>();
	public static ArrayList<GridLocation> allPits_treasure_X = new ArrayList<GridLocation>();
	
	
	public GridWorldDomainFeatures gwdf_treasure_X = new GridWorldDomainFeatures(MAP_treasure_X, new GridAgent(1,1),	new GridLocation(2,1,3,"treasure"),	allFires_treasure_X, allPits_treasure_X, tag_treasure_X);
	
	/** MAP_treasure_X_L
	 * 
	 * O O O O O O O O O 
	 * O A           T O
	 * O O O O O O O O O
	 * */
	
	public static final int[][] MAP_treasure_X_L = {{1,1,1},
													{1,0,1},
													{1,0,1},
													{1,0,1},
													{1,0,1},
													{1,0,1},
													{1,0,1},
													{1,0,1},
													{1,1,1}};
	
	public static String tag_treasure_X_L = "treasure_X_L";
	
	public static ArrayList<GridLocation> allFires_treasure_X_L = new ArrayList<GridLocation>();
	public static ArrayList<GridLocation> allPits_treasure_X_L = new ArrayList<GridLocation>();
	
	
	public GridWorldDomainFeatures gwdf_treasure_X_L = new GridWorldDomainFeatures(MAP_treasure_X_L, new GridAgent(1,1),	new GridLocation(7,1,3,"treasure"),	allFires_treasure_X_L, allPits_treasure_X_L, tag_treasure_X_L);
	
	/************************************** TREASURE MAPS ******************************************/
	
	/** MAP_treasure
	 * 
	 * O O O O O
	 * O       O
	 * O   T   O
	 * O A     O
	 * O O O O O
	 * */
	
	public static final int[][] MAP_treasure = {{1,1,1,1,1},
												{1,0,0,0,1},
												{1,0,0,0,1},
												{1,0,0,0,1},
												{1,1,1,1,1}};
	
	public static String tag_treasure = "treasure";
	
	public static ArrayList<GridLocation> allFires_treasure = new ArrayList<GridLocation>();
	public static ArrayList<GridLocation> allPits_treasure = new ArrayList<GridLocation>();
	
	
	public GridWorldDomainFeatures gwdf_treasure = new GridWorldDomainFeatures(MAP_treasure, new GridAgent(1,1), new GridLocation(2,2,3,"treasure"), allFires_treasure, allPits_treasure, tag_treasure);
	
	/** MAP_treasure_L
	 * 
	 * O O O O O O 
	 * O       T O
	 * O         O
	 * O         O
	 * O A       O
	 * O O O O O O
	 * */
	
	public static final int[][] MAP_treasure_L = {	{1,1,1,1,1,1},
													{1,0,0,0,0,1},
													{1,0,0,0,0,1},
													{1,0,0,0,0,1},
													{1,0,0,0,0,1},
													{1,1,1,1,1,1}};
	
	public static String tag_treasure_L = "treasure_L";
	
	public static ArrayList<GridLocation> allFires_treasure_L = new ArrayList<GridLocation>();
	public static ArrayList<GridLocation> allPits_treasure_L = new ArrayList<GridLocation>();
	
	
	public GridWorldDomainFeatures gwdf_treasure_L = new GridWorldDomainFeatures(MAP_treasure_L, new GridAgent(1,1), new GridLocation(4,4,3,"treasure"),	allFires_treasure_L, allPits_treasure_L, tag_treasure_L);
	
	/** MAP_treasure_XL
	 * 
	 * O O O O O O O O O O O 
	 * O                 T O
	 * O                   O
	 * O                   O
	 * O                   O
	 * O                   O
	 * O                   O
	 * O                   O
	 * O                   O
	 * O A                 O
	 * O O O O O O O O O O O
	 * */
	
	public static final int[][] MAP_treasure_XL = {	{1,1,1,1,1,1,1,1,1,1},
													{1,0,0,0,0,0,0,0,0,1},
													{1,0,0,0,0,0,0,0,0,1},
													{1,0,0,0,0,0,0,0,0,1},
													{1,0,0,0,0,0,0,0,0,1},
													{1,0,0,0,0,0,0,0,0,1},
													{1,0,0,0,0,0,0,0,0,1},
													{1,0,0,0,0,0,0,0,0,1},
													{1,0,0,0,0,0,0,0,0,1},
													{1,1,1,1,1,1,1,1,1,1}};
	
	public static String tag_treasure_XL = "treasure_XL";
	
	public static ArrayList<GridLocation> allFires_treasure_XL = new ArrayList<GridLocation>();
	public static ArrayList<GridLocation> allPits_treasure_XL = new ArrayList<GridLocation>();
	
	
	public GridWorldDomainFeatures gwdf_treasure_XL = new GridWorldDomainFeatures(MAP_treasure_XL, new GridAgent(1,1),	new GridLocation(8,8,3,"treasure"),	allFires_treasure_XL, allPits_treasure_XL, tag_treasure_XL);
	
	/************************************** TEST MAPS ******************************************/
	
	/** MAP_test
	 * 
	 * O O O O O O 
	 * O       T O
	 * O         O
	 * O   F     O
	 * O A P     O
	 * O O O O O O
	 * */
	
	public static final int[][] MAP_test = {{1,1,1,1,1,1},
											{1,0,0,0,0,1},
											{1,0,0,0,0,1},
											{1,0,0,0,0,1},
											{1,0,0,0,0,1},
											{1,1,1,1,1,1}};
	
	public static String tag_test = "test";
	
	public static ArrayList<GridLocation> allFires_test = new ArrayList<GridLocation>() {{add(new GridLocation(3,2,1,"fire1"));}};
	public static ArrayList<GridLocation> allPits_test = new ArrayList<GridLocation>() {{add(new GridLocation(2,1,0,"pit1"));}};
	
	
	public GridWorldDomainFeatures gwdf_test = new GridWorldDomainFeatures(MAP_test, new GridAgent(1,1),	new GridLocation(4,4,3,"treasure"),	allFires_test, allPits_test, tag_test);

	/** MAP_test1
	 * 
	 * O O O O O O 
	 * O F     T O
	 * O         O
	 * O       P O
	 * O A       O
	 * O O O O O O
	 * */
	
	public static String tag_test1 = "test1";
	
	public static ArrayList<GridLocation> allFires_test1 = new ArrayList<GridLocation>() {{add(new GridLocation(1,4,1,"fire1"));}};
	public static ArrayList<GridLocation> allPits_test1 = new ArrayList<GridLocation>() {{add(new GridLocation(4,2,0,"pit1"));}};
	
	
	public GridWorldDomainFeatures gwdf_test1 = new GridWorldDomainFeatures(MAP_test, new GridAgent(1,1),	new GridLocation(4,4,3,"treasure"),	allFires_test1, allPits_test1, tag_test1);
	
	/** MAP_test2
	 * 
	 * O O O O O O 
	 * O   F     O
	 * O         O
	 * O     P T O
	 * O A       O
	 * O O O O O O
	 * */
	
	public static String tag_test2 = "test2";
	
	public static ArrayList<GridLocation> allFires_test2 = new ArrayList<GridLocation>() {{add(new GridLocation(2,4,1,"fire1"));}};
	public static ArrayList<GridLocation> allPits_test2 = new ArrayList<GridLocation>() {{add(new GridLocation(3,2,0,"pit1"));}};
	
	
	public GridWorldDomainFeatures gwdf_test2 = new GridWorldDomainFeatures(MAP_test, new GridAgent(1,1),	new GridLocation(4,2,3,"treasure"),	allFires_test2, allPits_test2, tag_test2);
	
	/** MAP_test_manyF
	 * 
	 * O O O O O O 
	 * O   F   T O
	 * O         O
	 * O       F O
	 * O A   P   O
	 * O O O O O O
	 * */
	
	/** This task results quite difficult for finding its optimal policy. 
	 *  A different agent or parameter values could solve the issue */
	
	public static String tag_test_manyF = "test_manyF";
	
	public static ArrayList<GridLocation> allFires_test_manyF = new ArrayList<GridLocation>() {{add(new GridLocation(4,2,1,"fire1"));
																								add(new GridLocation(1,4,1,"fire2"));}};
	public static ArrayList<GridLocation> allPits_test_manyF = new ArrayList<GridLocation>() {{add(new GridLocation(3,1,0,"pit1"));}};
	
	
	public GridWorldDomainFeatures gwdf_test_manyF = new GridWorldDomainFeatures(MAP_test, new GridAgent(1,1),	new GridLocation(4,4,3,"treasure"),	allFires_test_manyF, allPits_test_manyF, tag_test_manyF);
	
	
	/** MAP_test_M
	 * 
	 * O O O O O O O O 
	 * O A           O
	 * O   P         O
	 * O             O
	 * O           P O
	 * O F F         O
	 * O           T O
	 * O O O O O O O O
	 * */
	
	public static final int[][] MAP_test_M = {{1,1,1,1,1,1,1,1},
										{1,0,0,0,0,0,0,1},
										{1,0,0,0,0,0,0,1},
										{1,0,0,0,0,0,0,1},
										{1,0,0,0,0,0,0,1},
										{1,0,0,0,0,0,0,1},
										{1,0,0,0,0,0,0,1},
										{1,1,1,1,1,1,1,1}};
	
	public static String tag_test_M = "test_M";
	
	public static ArrayList<GridLocation> allFires_test_M = new ArrayList<GridLocation>() {{add(new GridLocation(1,2,1,"fire1"));
																							add(new GridLocation(2,2,1,"fire2"));}};
	public static ArrayList<GridLocation> allPits_test_M = new ArrayList<GridLocation>() {{add(new GridLocation(6,3,0,"pit1"));
																						add(new GridLocation(2,5,0,"pit2"));}};;
	
	
	public GridWorldDomainFeatures gwdf_test_M = new GridWorldDomainFeatures(MAP_test_M, new GridAgent(1,6), new GridLocation(6,1,3,"treasure"), allFires_test_M, allPits_test_M, tag_test_M);
	
	/** MAP_test_M_1
	 * 
	 * O O O O O O O O 
	 * O             O
	 * O   F       P O
	 * O   T     F   O
	 * O       P     O
	 * O             O
	 * O A           O
	 * O O O O O O O O
	 * */
	
    public static String tag_test_M_1 = "test_M_1";
	
	public static ArrayList<GridLocation> allFires_test_M_1 = new ArrayList<GridLocation>() {{add(new GridLocation(5,4,1,"fire1"));
																							add(new GridLocation(2,5,1,"fire2"));}};
	public static ArrayList<GridLocation> allPits_test_M_1 = new ArrayList<GridLocation>() {{add(new GridLocation(4,3,0,"pit1"));
																						add(new GridLocation(6,5,0,"pit2"));}};;
	
	
	public GridWorldDomainFeatures gwdf_test_M_1 = new GridWorldDomainFeatures(MAP_test_M, new GridAgent(1,1), new GridLocation(2,4,3,"treasure"), allFires_test_M_1, allPits_test_M_1, tag_test_M_1);
		
	/** MAP_test_M_2
	 * 
	 * O O O O O O O O 
	 * O F         T O
	 * O             O
	 * O           P O
	 * O       F     O
	 * O     P       O
	 * O A           O
	 * O O O O O O O O
	 * */
	
    public static String tag_test_M_2 = "test_M_2";
	
	public static ArrayList<GridLocation> allFires_test_M_2 = new ArrayList<GridLocation>() {{add(new GridLocation(4,3,1,"fire1"));
																							add(new GridLocation(1,6,1,"fire2"));}};
	public static ArrayList<GridLocation> allPits_test_M_2 = new ArrayList<GridLocation>() {{add(new GridLocation(3,2,0,"pit1"));
																						add(new GridLocation(6,4,0,"pit2"));}};;
	
	
	public GridWorldDomainFeatures gwdf_test_M_2 = new GridWorldDomainFeatures(MAP_test_M, new GridAgent(1,1), new GridLocation(6,6,3,"treasure"), allFires_test_M_2, allPits_test_M_2, tag_test_M_2);
		
	/** MAP_test_M_manyP
	 * 
	 * O O O O O O O O 
	 * O A         F O
	 * O           P O
	 * O         F   O
	 * O P     P     O
	 * O   P         O
	 * O           T O
	 * O O O O O O O O
	 * */
	
    public static String tag_test_M_manyP = "test_M_manyP";
	
	public static ArrayList<GridLocation> allFires_test_M_manyP = new ArrayList<GridLocation>() {{add(new GridLocation(5,4,1,"fire1"));
																							add(new GridLocation(6,6,1,"fire2"));}};
	public static ArrayList<GridLocation> allPits_test_M_manyP = new ArrayList<GridLocation>() {{add(new GridLocation(2,2,0,"pit1"));
																								add(new GridLocation(1,3,0,"pit2"));
																								add(new GridLocation(4,3,0,"pit3"));
																								add(new GridLocation(6,5,0,"pit4"));}};;
	
	
	public GridWorldDomainFeatures gwdf_test_M_manyP = new GridWorldDomainFeatures(MAP_test_M, new GridAgent(1,6), new GridLocation(6,1,3,"treasure"), allFires_test_M_manyP, allPits_test_M_manyP, tag_test_M_manyP);
	
	
	/** MAP_test_L
	 * 
	 * O O O O O O O O O O O 
	 * O                 T O
	 * O                 P O
	 * O                   O
	 * O             F     O
	 * O                   O
	 * O                   O
	 * O                   O
	 * O     F             O
	 * O A P               O
	 * O O O O O O O O O O O
	 * */
	
	public static String tag_test_L = "test_L";
	
	public static ArrayList<GridLocation> allFires_test_L = new ArrayList<GridLocation>() {{add(new GridLocation(3,2,1,"fire1"));
																							add(new GridLocation(7,6,1,"fire2"));}};
	public static ArrayList<GridLocation> allPits_test_L = new ArrayList<GridLocation>() {{add(new GridLocation(2,1,0,"pit1"));
																							add(new GridLocation(8,7,0,"pit2"));}};;
	
	
	public GridWorldDomainFeatures gwdf_test_L = new GridWorldDomainFeatures(MAP_treasure_XL, new GridAgent(1,1),	new GridLocation(8,8,3,"treasure"),	allFires_test_L, allPits_test_L, tag_test_L);
	
	/** MAP_test_Lv2
	 * 
	 * O O O O O O O O O O O 
	 * O                   O
	 * O           T       O
	 * O               F   O
	 * O           P       O
	 * O                   O
	 * O                   O
	 * O     F             O
	 * O                   O
	 * O A               P O
	 * O O O O O O O O O O O
	 * */
	
	public static String tag_test_Lv2 = "test_Lv2";
	
	public static ArrayList<GridLocation> allFires_test_Lv2 = new ArrayList<GridLocation>() {{add(new GridLocation(3,3,1,"fire1"));
																							add(new GridLocation(7,6,1,"fire2"));}};
	public static ArrayList<GridLocation> allPits_test_Lv2 = new ArrayList<GridLocation>() {{add(new GridLocation(8,1,0,"pit1"));
																							add(new GridLocation(5,5,0,"pit2"));}};;
	
	
	public GridWorldDomainFeatures gwdf_test_Lv2 = new GridWorldDomainFeatures(MAP_treasure_XL, new GridAgent(1,1),	new GridLocation(5,7,3,"treasure"),	allFires_test_Lv2, allPits_test_Lv2, tag_test_Lv2);
	
	/** MAP_test_L_2
	 * 
	 * O O O O O O O O O O O 
	 * O T       P         O
	 * O               P   O
	 * O F F               O
	 * O                   O
	 * O                   O
	 * O   P         F     O
	 * O                 F O
	 * O                   O
	 * O       P         A O
	 * O O O O O O O O O O O
	 * */
	
    public static String tag_test_L_2 = "test_L_2";
	
	public static ArrayList<GridLocation> allFires_test_L_2 = new ArrayList<GridLocation>() {{add(new GridLocation(8,3,1,"fire1"));
																							add(new GridLocation(6,4,1,"fire2"));
																							add(new GridLocation(1,6,1,"fire3"));
																							add(new GridLocation(2,6,1,"fire4"));}};
	public static ArrayList<GridLocation> allPits_test_L_2 = new ArrayList<GridLocation>() {{add(new GridLocation(4,1,0,"pit1"));
																							add(new GridLocation(2,4,0,"pit2"));
																							add(new GridLocation(7,7,0,"pit3"));
																							add(new GridLocation(5,8,0,"pit4"));}};;
	
	
	public GridWorldDomainFeatures gwdf_test_L_2 = new GridWorldDomainFeatures(MAP_treasure_XL, new GridAgent(8,1),	new GridLocation(1,8,3,"treasure"),	allFires_test_L_2, allPits_test_L_2, tag_test_L_2);
		
	
	/** MAP_test_Lv2_2
	 * 
	 * O O O O O O O O O O O 
	 * O                 P O
	 * O   P               O
	 * O F           F     O
	 * O                   O
	 * O           T       O
	 * O         P   F     O
	 * O F                 O
	 * O                   O
	 * O       P         A O
	 * O O O O O O O O O O O
	 * */
	
    public static String tag_test_Lv2_2 = "test_Lv2_2";
	
	public static ArrayList<GridLocation> allFires_test_Lv2_2 = new ArrayList<GridLocation>() {{add(new GridLocation(1,3,1,"fire1"));
																							add(new GridLocation(7,4,1,"fire2"));
																							add(new GridLocation(1,6,1,"fire3"));
																							add(new GridLocation(7,6,1,"fire4"));}};
	public static ArrayList<GridLocation> allPits_test_Lv2_2 = new ArrayList<GridLocation>() {{add(new GridLocation(4,1,0,"pit1"));
																							add(new GridLocation(4,6,0,"pit2"));
																							add(new GridLocation(2,7,0,"pit3"));
																							add(new GridLocation(8,8,0,"pit4"));}};;
	
	
	public GridWorldDomainFeatures gwdf_test_Lv2_2 = new GridWorldDomainFeatures(MAP_treasure_XL, new GridAgent(8,1),	new GridLocation(5,5,3,"treasure"),	allFires_test_Lv2_2, allPits_test_Lv2_2, tag_test_Lv2_2);
		
	/** MAP_test_L_manyP
	 * 
	 * O O O O O O O O O O O 
	 * O T   P             O
	 * O P             F   O
	 * O F       P   P     O
	 * O      F            O
	 * O             P     O
	 * O         P       F O
	 * O         P         O
	 * O     P           A O
	 * O O O O O O O O O O O
	 * */
	
	/** This task results quite difficult to solve. It could be an interesting Final Task 
	 *  or it would probably need a bigger function approximator in order to effciently 
	 *  find a solution */
	
    public static String tag_test_L_manyP = "test_L_manyP";
	
	public static ArrayList<GridLocation> allFires_test_L_manyP = new ArrayList<GridLocation>() {{add(new GridLocation(8,3,1,"fire1"));
																							add(new GridLocation(1,5,1,"fire2"));
																							add(new GridLocation(3,5,1,"fire3"));
																							add(new GridLocation(7,7,1,"fire4"));}};
	public static ArrayList<GridLocation> allPits_test_L_manyP = new ArrayList<GridLocation>() {{add(new GridLocation(2,1,0,"pit1"));
																							add(new GridLocation(4,2,0,"pit2"));
																							add(new GridLocation(4,3,0,"pit3"));
																							add(new GridLocation(6,4,0,"pit4"));
																							add(new GridLocation(1,6,0,"pit5"));
																							add(new GridLocation(4,6,0,"pit6"));
																							add(new GridLocation(6,6,0,"pit7"));
																							add(new GridLocation(3,8,0,"pit8"));}};;
	
	
	public GridWorldDomainFeatures gwdf_test_L_manyP = new GridWorldDomainFeatures(MAP_treasure_XL, new GridAgent(8,1),	new GridLocation(1,8,3,"treasure"),	allFires_test_L_manyP, allPits_test_L_manyP, tag_test_L_manyP);
	
	/** MAP_fire
	 * 
	 * 
	 * O O O O O
	 * O     T O
	 * O   F   O
	 * O A     O
	 * O O O O O
	 * */
	
	public static final int[][] MAP_fire = {{1,1,1,1,1},
											{1,0,0,0,1},
											{1,0,0,0,1},
											{1,0,0,0,1},
											{1,1,1,1,1}};
	
	public static String tag_fire = "fire";
	
	public static ArrayList<GridLocation> allFires_fire = new ArrayList<GridLocation>() {{add(new GridLocation(2,3,1,"fire1"));}};
	public static ArrayList<GridLocation> allPits_fire = new ArrayList<GridLocation>();
	
	
	public GridWorldDomainFeatures gwdf_fire = new GridWorldDomainFeatures(MAP_fire, new GridAgent(1,1), new GridLocation(3,3,3,"treasure"), allFires_fire, allPits_fire, tag_fire);
	
	/** MAP_pit
	 * 
	 * 
	 * O O O O O
	 * O     T O
	 * O   P   O
	 * O A     O
	 * O O O O O
	 * */
	
	public static final int[][] MAP_pit = { {1,1,1,1,1},
											{1,0,0,0,1},
											{1,0,0,0,1},
											{1,0,0,0,1},
											{1,1,1,1,1}};
	
	public static String tag_pit = "pit";

	
	public static ArrayList<GridLocation> allFires_pit = new ArrayList<GridLocation>();
	public static ArrayList<GridLocation> allPits_pit = new ArrayList<GridLocation>() {{add(new GridLocation(2,2,0,"pit1"));}};
	
	
	public GridWorldDomainFeatures gwdf_pit = new GridWorldDomainFeatures(MAP_pit, new GridAgent(1,1), new GridLocation(3,3,3,"treasure"), allFires_pit, allPits_pit, tag_pit);
	
	/** MAP_0
	 * 
	 * O O O O O O O O O O O O
	 * O A                   O
	 * O                     O
	 * O F F F F P   P P F F O
	 * O				     O 
	 * O   F     P	 P     F O 
	 * O 		   F	     O 
	 * O				     O
	 * O           P   P F F O
	 * O                     O
	 * O             P     T O
	 * O O O O O O O O O O O O
	 * */
	
	/** Very difficult to learn but definitely an optimal candidate for being a Final Task
	 *  as it was also used in other publications */
	
	public static final int[][] MAP_0 = {{1,1,1,1,1,1,1,1,1,1,1,1},
										{1,0,0,0,0,0,0,0,0,0,0,1},
										{1,0,0,0,0,0,0,0,0,0,0,1},
										{1,0,0,0,0,0,0,0,0,0,0,1},
										{1,0,0,0,0,0,0,0,0,0,0,1},
										{1,0,0,0,0,0,0,0,0,0,0,1},
										{1,0,0,0,0,0,0,0,0,0,0,1},
										{1,0,0,0,0,0,0,0,0,0,0,1},
										{1,0,0,0,0,0,0,0,0,0,0,1},
										{1,0,0,0,0,0,0,0,0,0,0,1},
										{1,0,0,0,0,0,0,0,0,0,0,1},
										{1,1,1,1,1,1,1,1,1,1,1,1}};
	
	public static String tag_0 = "0";
	
	public static ArrayList<GridLocation> allFires_0 = new ArrayList<GridLocation>() {{add(new GridLocation(9,3,1,"fire1"));
																					add(new GridLocation(10,3,1,"fire2"));
																					add(new GridLocation(6,5,1,"fire3"));
																					add(new GridLocation(2,6,1,"fire4"));
																					add(new GridLocation(10,6,1,"fire5"));
																					add(new GridLocation(1,8,1,"fire6"));
																					add(new GridLocation(2,8,1,"fire7"));
																					add(new GridLocation(3,8,1,"fire8"));
																					add(new GridLocation(4,8,1,"fire9"));
																					add(new GridLocation(9,8,1,"fire10"));
																					add(new GridLocation(10,8,1,"fire11"));}};
																					
	public static ArrayList<GridLocation> allPits_0 = new ArrayList<GridLocation>() {{add(new GridLocation(7,1,0,"pit1"));
																					add(new GridLocation(6,3,0,"pit2"));
																					add(new GridLocation(8,3,0,"pit3"));
																					add(new GridLocation(5,6,0,"pit4"));
																					add(new GridLocation(7,6,0,"pit5"));
																					add(new GridLocation(5,8,0,"pit6"));
																					add(new GridLocation(7,8,0,"pit7"));
																					add(new GridLocation(8,8,0,"pit8"));}};
	
	
	public GridWorldDomainFeatures gwdf_0 = new GridWorldDomainFeatures(MAP_0, new GridAgent(1,10), new GridLocation(10,1,3,"treasure"), allFires_0, allPits_0, tag_0);
	
	
	public class GridWorldDomainFeatures {
		
		public int[][] map;
		public GridAgent agent;
		public GridLocation treasure;
		public ArrayList<GridLocation> fires;
		public ArrayList<GridLocation> pits;
		public String tag;
		
		GridWorldDomainFeatures(int[][] m, GridAgent ga, GridLocation t, ArrayList<GridLocation> fs, ArrayList<GridLocation> ps, String tg){
			
			this.map = m;
			this.agent = ga;
			this.treasure = t;
			this.fires = fs;
			this.pits = ps;
			this.tag = tg;
			
		}
		
		public GridWorldState generateInitialState() {
			return new GridWorldState(this.agent, new ArrayList<GridLocation>() {{add(treasure); addAll(fires); addAll(pits); }});
		}
		
	}
	
	/** Function for writing the action-values relative to the specified state s on file*/
	public static void WriteActionValues(PrintWriter writer, QProvider qp, State s, long ep){
		writer.print(ep + " ");
		List<QValue> sQValues = qp.qValues(s);
		for(ListIterator<QValue> itr = sQValues.listIterator(); itr.hasNext() ;){
			QValue currentQValue = itr.next();
			double actionValue = currentQValue.q;
			
			writer.print(actionValue+ " ");
		}
		writer.println();
	}
	
	/** Function for performing and saving the result of the execution of the greedy policy over the learnt Action-Value Function
	 *  @param episode is: <0 when we just want to print the execution of the episode, >0 when we want to have the reward file*/
	public static double PerformLearntPolicy(GradientDescentSarsaLam agentRef, SimulatedEnvironment environment, String outputPath, String name, long episode){
		
		/** modify the agent for fully not learning behaviours */
		agentRef.setLearningPolicy(new GreedyDeterministicQPolicy(agentRef));
		agentRef.setLearningRate(new ConstantLR(0.0)); 
		
		Episode e = agentRef.runLearningEpisode(environment,50);
		
		double r = e.discountedReturn(1); 
		double a = agentRef.getLastNumSteps(); 
		
		if(episode >= 0){
			BufferedWriter buff = null;
			try {
				FileWriter osPE = new FileWriter(outputPath+name, true);
				buff = new BufferedWriter(osPE);
				buff.write("\n" + episode + " " + r + " " + a);
				buff.close();
			} catch (IOException e1) {
				System.err.println("Error: " + e1.getMessage());
			}
		}else	
			e.write(outputPath + "_" + name);
		
		environment.resetEnvironment();
		
		return r;
	}
	
	
	
	public static TLTileCodingFeatures CreateTransferLearningTCFeatures(String tag, int numTilings) {
		
		TLTileCodingFeatures tileCoding = null;
		
		if(tag.equals(tag_test)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(0,3));	domains.put("treasure:y", new VariableDomainFloatingNorm(0,3));
			domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));	domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
			
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			tilingWidths[4] = 0.25; tilingWidths[5] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(tag.equals(tag_test1)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(0,3));domains.put("treasure:y", new VariableDomainFloatingNorm(0,3));
			domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
			
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			tilingWidths[4] = 0.25; tilingWidths[5] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(tag.equals(tag_test2)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(0,3));domains.put("treasure:y", new VariableDomainFloatingNorm(-2,1));
			domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
			
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			tilingWidths[4] = 0.25; tilingWidths[5] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(tag.equals(tag_test_manyF)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(0,3));domains.put("treasure:y", new VariableDomainFloatingNorm(0,3));
			domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relFire2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire2:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
			
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			tilingWidths[4] = 0.25; tilingWidths[5] = 0.25;
			tilingWidths[6] = 0.25; tilingWidths[7] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(tag.equals(tag_test_M)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(0,5));domains.put("treasure:y", new VariableDomainFloatingNorm(-5,0));
			domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relFire2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire2:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
			domains.put("relPit2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit2:y", new VariableDomainFloatingNorm(-2,2));
			
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			tilingWidths[4] = 0.25; tilingWidths[5] = 0.25;
			tilingWidths[6] = 0.25; tilingWidths[7] = 0.25;
			tilingWidths[8] = 0.25; tilingWidths[9] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(tag.equals(tag_test_M_1)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(-4,1));domains.put("treasure:y", new VariableDomainFloatingNorm(-2,3));
			domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relFire2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire2:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
			domains.put("relPit2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit2:y", new VariableDomainFloatingNorm(-2,2));
			
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			tilingWidths[4] = 0.25; tilingWidths[5] = 0.25;
			tilingWidths[6] = 0.25; tilingWidths[7] = 0.25;
			tilingWidths[8] = 0.25; tilingWidths[9] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(tag.equals(tag_test_M_2)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(0,5));domains.put("treasure:y", new VariableDomainFloatingNorm(0,5));
			domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relFire2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire2:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
			domains.put("relPit2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit2:y", new VariableDomainFloatingNorm(-2,2));
			
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			tilingWidths[4] = 0.25; tilingWidths[5] = 0.25;
			tilingWidths[6] = 0.25; tilingWidths[7] = 0.25;
			tilingWidths[8] = 0.25; tilingWidths[9] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(tag.equals(tag_test_M_manyP)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(0,5));domains.put("treasure:y", new VariableDomainFloatingNorm(-5,0));
			domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relFire2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire2:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
			domains.put("relPit2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit2:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit3:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit3:y", new VariableDomainFloatingNorm(-2,2));
			
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			tilingWidths[4] = 0.25; tilingWidths[5] = 0.25;
			tilingWidths[6] = 0.25; tilingWidths[7] = 0.25;
			tilingWidths[8] = 0.25; tilingWidths[9] = 0.25;
			tilingWidths[10] = 0.25; tilingWidths[11] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(tag.equals(tag_test_L)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(0,7));domains.put("treasure:y", new VariableDomainFloatingNorm(0,7));
			domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relFire2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire2:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
			domains.put("relPit2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit2:y", new VariableDomainFloatingNorm(-2,2));
			
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			tilingWidths[4] = 0.25; tilingWidths[5] = 0.25;
			tilingWidths[6] = 0.25; tilingWidths[7] = 0.25;
			tilingWidths[8] = 0.25; tilingWidths[9] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(tag.equals(tag_test_Lv2)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(-3,4));domains.put("treasure:y", new VariableDomainFloatingNorm(-1,6));
			domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relFire2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire2:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
			domains.put("relPit2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit2:y", new VariableDomainFloatingNorm(-2,2));
			
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			tilingWidths[4] = 0.25; tilingWidths[5] = 0.25;
			tilingWidths[6] = 0.25; tilingWidths[7] = 0.25;
			tilingWidths[8] = 0.25; tilingWidths[9] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(tag.equals(tag_test_L_2)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(-7,0));domains.put("treasure:y", new VariableDomainFloatingNorm(0,7));
			domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relFire2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire2:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
			domains.put("relPit2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit2:y", new VariableDomainFloatingNorm(-2,2));
			
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			tilingWidths[4] = 0.25; tilingWidths[5] = 0.25;
			tilingWidths[6] = 0.25; tilingWidths[7] = 0.25;
			tilingWidths[8] = 0.25; tilingWidths[9] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(tag.equals(tag_test_Lv2_2)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(-3,4));domains.put("treasure:y", new VariableDomainFloatingNorm(-3,4));
			domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relFire2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire2:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
			domains.put("relPit2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit2:y", new VariableDomainFloatingNorm(-2,2));
			
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			tilingWidths[4] = 0.25; tilingWidths[5] = 0.25;
			tilingWidths[6] = 0.25; tilingWidths[7] = 0.25;
			tilingWidths[8] = 0.25; tilingWidths[9] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(tag.equals(tag_test_L_manyP)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(-7,0));domains.put("treasure:y", new VariableDomainFloatingNorm(0,7));
			domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relFire2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire2:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
			domains.put("relPit2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit2:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit3:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit3:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit4:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit4:y", new VariableDomainFloatingNorm(-2,2));
			
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			tilingWidths[4] = 0.25; tilingWidths[5] = 0.25;
			tilingWidths[6] = 0.25; tilingWidths[7] = 0.25;
			tilingWidths[8] = 0.25; tilingWidths[9] = 0.25;
			tilingWidths[10] = 0.25; tilingWidths[11] = 0.25;
			tilingWidths[12] = 0.25; tilingWidths[12] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(tag.equals(tag_treasure_X)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();	
			domains.put("treasure:x", new VariableDomainFloatingNorm(0,1));	domains.put("treasure:y", new VariableDomainFloatingNorm(0,0));	
			
			TLNormalizedVarFeatGridWorld stateVariables = new TLNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 1;	tilingWidths[1] = 0;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
		}else if(tag.equals(tag_treasure_X_L)) {
		    Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(0,7)); domains.put("treasure:y", new VariableDomainFloatingNorm(0,0));	
			
			TLNormalizedVarFeatGridWorld stateVariables = new TLNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.25;	tilingWidths[1] = 0;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
		}else if(tag.equals(tag_treasure)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();			
			domains.put("treasure:x", new VariableDomainFloatingNorm(-1,1));domains.put("treasure:y", new VariableDomainFloatingNorm(-1,1));	
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
		}else if(tag.equals(tag_treasure_L)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(0,3));domains.put("treasure:y", new VariableDomainFloatingNorm(0,3));			
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
		}else if(tag.equals(tag_treasure_XL)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomain(0,8));domains.put("treasure:y", new VariableDomain(0,8));			
			
			TLNormalizedVarFeatGridWorld stateVariables = new TLNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.25; tilingWidths[1] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
		}else if(tag.equals(tag_fire)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(0,2));domains.put("treasure:y", new VariableDomainFloatingNorm(0,2));
			domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
		}else if(tag.equals(tag_pit)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(0,2));domains.put("treasure:y", new VariableDomainFloatingNorm(0,2));
			domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
		}else if(tag.equals(tag_0)) {
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("treasure:x", new VariableDomainFloatingNorm(0,9));domains.put("treasure:y", new VariableDomainFloatingNorm(-9,0));
			domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relFire2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire2:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relFire3:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire3:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relFire4:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire4:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relFire5:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire5:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
			domains.put("relPit2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit2:y", new VariableDomainFloatingNorm(-2,2));	
			domains.put("relPit3:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit3:y", new VariableDomainFloatingNorm(-2,2));	
			domains.put("relPit4:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit4:y", new VariableDomainFloatingNorm(-2,2));	
			
			TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];

			tilingWidths[0] = 0.1; tilingWidths[1] = 0.1;
			tilingWidths[2] = 0.25; tilingWidths[3] = 0.25;
			tilingWidths[4] = 0.25; tilingWidths[5] = 0.25;
			tilingWidths[6] = 0.25; tilingWidths[7] = 0.25;
			tilingWidths[8] = 0.25; tilingWidths[9] = 0.25;
			tilingWidths[10] = 0.25; tilingWidths[11] = 0.25;
			tilingWidths[12] = 0.25; tilingWidths[13] = 0.25;
			tilingWidths[14] = 0.25; tilingWidths[15] = 0.25;
			tilingWidths[16] = 0.25; tilingWidths[17] = 0.25;
			tilingWidths[18] = 0.25; tilingWidths[19] = 0.25;
			
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
		}else
			System.err.println(tag + " is not implemented yet.");
		
		return tileCoding;
		
	}
	
}

package CurriculumLearning;

import TransferLearning.*;
import BlockDude.*;

import java.io.File;
import java.util.ArrayList;

import CurriculumLearning.CurriculumLearning_UTILS.CurriculumGenerator;
import GridWorld.GridWorldTransferLearning;
import GridWorld.GridWorldTransferLearning_UTILS;
import GridWorld.GridWorldTransferLearning_UTILS.GridWorldDomainFeatures;
import burlap.domain.singleagent.blockdude.state.BlockDudeState;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.state.State;

public class CurriculumLearning {
		
		ArrayList<CurriculumStep> 		sourceTasks;
		CurriculumStep 					targetTask;
			
		public CurriculumLearning(ArrayList<CurriculumStep> sources, CurriculumStep target){
			
			this.sourceTasks = sources;
			this.targetTask = target;
			
		};
		
		public void performCurriculum(String outputFolder){
			
			TLLinearVFA allPreviousVFA = this.fullCascadeCurriculumLearning(outputFolder);
			
			this.targetTask.learn(outputFolder + "outputTarget/", allPreviousVFA);
			this.targetTask.runVisualizer();
			
		};	
		
		/** The set of source tasks is used to create and run a curriculum using
		 *  all of the sources in the given order */
		private TLLinearVFA fullCascadeCurriculumLearning(String outputFolder){
			
			TLLinearVFA previousVFA = null;
			int i = 0;
			
			for(CurriculumStep currSource : this.sourceTasks){
				System.out.println("Source " + i);
				
				String currOut = outputFolder + "outputSource" + i + "/";
				
				previousVFA = currSource.learn(currOut, previousVFA);
				currSource.runVisualizer();
				
				++i;
			}
			
			return previousVFA;
		}
		
		/************************* Curriculum Step Class ****************************
		 * The class wraps together all the useful information characterizing a source 
		 * task in the curriculum */
		
		
		public static class CurriculumStep{
			
			Object							taskState;
			int 							numTilings;
			int 							numEpisodes;
			
			TransferLearning				taskTransferLearning 	= null;//TODO
			String							outputPath   			= null;
			
			/** Basic Constructor */
			public CurriculumStep(Object source){
				this.taskState = source;
				
				int[] supp = CurriculumLearning_UTILS.getExeParamsForLearning(this.taskState);
				
				this.numTilings = supp[0];
				this.numEpisodes = supp[1];
			}
			
			/** Creation of the TransferLearning object with which you can learn the given task */
			private void initTransferLearning(String outputPath){
				
				this.outputPath = outputPath;
				BlockDudeTransferLearning_UTILS.delete(new File(outputPath));//TODO
				
				if(this.taskState instanceof BlockDudeState)
					this.taskTransferLearning = new BlockDudeTransferLearning((BlockDudeState)this.taskState, this.numTilings);//TODO
				else if(this.taskState instanceof GridWorldDomainFeatures)
					this.taskTransferLearning = new GridWorldTransferLearning((GridWorldDomainFeatures)this.taskState, this.numTilings);
				else 
					System.err.println("ERROR: Type of variable taskState doesn't match any of the available types!");
			}
			
			public TLLinearVFA learn(String outputPath, TLLinearVFA sourceVFA){
				
				this.initTransferLearning(outputPath);
				
				return this.taskTransferLearning.PerformLearningAlgorithm(this.outputPath, 1, this.numEpisodes, sourceVFA);
				
			}
			
			public void runVisualizer(){
				if(this.outputPath != null)
					this.taskTransferLearning.visualize(this.outputPath);
				else
					System.out.println("The Output Path has not been initialized yet therefore you can't visualize the Learning Phase");
			}
		}
		
		/******************************    END    **********************************/
		
		/** Here you can run your curriculum experiments */
		public static void main(String[] args){
			
			/** EXPERIMENT 3 from paper */
			ArrayList<Object> sources = new ArrayList<Object>();
			
			/** Set of source taks. The order is not important here */
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test);		//0
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test1);		//1
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test2);		//2
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_L);		//3
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_Lv2);	//4
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_treasure);	//5
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_fire);		//6
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_pit);		//7
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_M);		//8
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_M_1);	//9
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_M_2);	//10
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_Lv2_2);	//11
			
			/** Instantiate a CurriculumGenerator by feeding the Sources, the Target Task, and the folder where we want to save the results */
			CurriculumGenerator gen = new CurriculumGenerator(sources, new GridWorldTransferLearning_UTILS().gwdf_treasure, "temp/EXP/");
			
			/** 
			 *  There are essentially two main ways to employ Curriculum Learning with this code:
			 *  
			 *  1) If generating curriculum from file you need to start with the first line and commenting the second one. 
			 *	Once you have done that you need to do vice versa for running a specific curriculum (the one at the line 
			 *	specified by the last argument of performCurriculumFromFile on the file you created at the previous step).
			 *
			 *  2) If you wish to run just one curriculum the only information you need is the number of epochs you want to
			 *  run it for and the ArrayList describing it. See the example for a better understanding.
			 ***********************************************************************
			 *   __ 
			 *  /_ |
			 *   | |
			 *   | |
			 *   | |
			 *   |_|
			 */     
			
			//gen.createAllCurriculaOnFile(0, 4, "temp/EXP/CurriculaDescription");
			//gen.performCurriculumFromFile(3, "temp/EXP/CurriculaDescription", 78);
			
			/** Alternative to the previous line if running a Task Array on Cluster */
			//gen.performCurriculumFromFile(10, "/nobackup/scff/burlap/CurriculaDescription", Integer.parseInt(System.getenv("SGE_TASK_ID"))-1);
			
			/***********************************************************************
			 *  ___  
			 * |__ \ 
			 *    ) |
			 *   / / 
			 *  / /_ 
			 * |____|
			 */
			
			gen.performCurriculumFromArray(10, new ArrayList<Integer> () {{
																			add(6);
																			add(9);
																			add(8);
																			}});
			
			/***********************************************************************/
			
			/** Printing Performance here*/
			gen.currPA.init();
			
			System.out.println("Max Return:\t\t" + gen.currPA.maxReturnStats());
			System.out.println("Cumulative Return:\t" + gen.currPA.cumulativeReturnStats());
			System.out.println("Jump Start:\t\t" + gen.currPA.jumpStartStats());
			System.out.println("Weak TTT:\t\t" + gen.currPA.weakTimeToThresholdStats(180));
			System.out.println("Strong TTT:\t\t" + gen.currPA.strongTimeToThresholdStats(180));
			
			//System.out.println("FINISHED!"); //Sometimes it can be useful
			
			return;
			
		}
}

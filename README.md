Software for experimenting the effects of Curriculum Learning on some simple domains and problems.

The whole project is based on the reinforceement learning library burlap (http://burlap.cs.brown.edu/).

# Set Up

## With IDE (Eclipse)
The following instructions show how to set up the Curriculum Learning Library (CLL) within the IDE Eclipse. You will need to use Maven for doing so and this might not be included in your IDE by default. If this is not the case, start reading from point number 1). 

0. Install Maven: 
    * Go to Help->Install New Software
    * In section "Work with" select "--All Available sites--"
    * Type in the text box "m2e" and then select "m2e - Maven Integration for Eclipse"
    * Finish installlation
1. Create a new maven project:
    * Go to: File->New->Other
    * Open th Maven folder and select Maven Project
    * Finish project creation (no specific options need to be selected for this project)
2. Copy the Curriculum Learning Library files into the project:
    * Download this project on your computer
    * Copy all the files into the newly created project (ProjectName/src/main/java)
    * You will see now all the Curriculum Learning Library files into the Package Explorer in your IDE (they will all show errors but that is normal as we still miss one step)
3. Edit Dependencies:
    * Open the pom.xml file in your maven project
    * Add a new dependency (Group Id = edu.brown.cs.burlap; Artifact Id = burlap; Version = 3.0.0)
    * Refresh your project

Your project now should not show any burlap-related error anymore.

** NOTE **: Some errors might still be present in the CLL as this project is not completed yet. Avoid using files with errors and ignore IDE messages about their existence. Limit the use of this code to Transfer Learning and Curriculum Learning experiments in the GridWorld domain.

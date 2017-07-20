#include<stdio.h>
#include<stdlib.h>




int main()
{

	int my_array[121];	
	int i;
	int n=0;
	FILE *fp;
	char output[]="output.txt";
	fp=fopen(output,"w");

	srand(0);

	for (i = 0; i <288; i++) 
	{


		fprintf(fp,"%s\t","[[");
		
		for(n=0;n<121;n++)
		{			
			my_array[n] = rand()%10;
			fprintf(fp,"%d",my_array[n]);
		}

		fprintf(fp,"%s\t\n\n","]]");

	//	fprintf("\n\n");
    		
	}

		

	fclose(fp);

}





















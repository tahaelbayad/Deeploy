
#include "DeeploySnitchMath.h"
#include "FloatAdd.h"

void SnitchFloatAdd(float32_t *pIn1, float32_t *pIn2, float32_t *pOut, uint32_t size) 
{
  uint32_t core_id = snrt_global_core_idx();
  if(core_id ==0 )
  {
    // for(uint32_t i = 0 ; i<16; i++)
    // {
    //   printf("dentro add in [%d] 1 = %f 2 = %f \n", i, pIn1[i], pIn2[i]);
    // }

    for (uint32_t i=0 ;i<size; i++){
      pOut[i] = pIn1[i] + pIn2[i];
      
      
    }

    // for(uint32_t i = 0 ; i<16; i++)
    // {
    //   printf("out add [%d] = %f  \n", i, pOut[i] );
    // }
  }   

}


#include "include/Random123/philox.h"
#include <stdio.h>

int main(int argc, char **argv){
    int i;
    //threefry2x64_ctr_t  ctr = {{0,0}};
    //threefry2x64_key_t key = {{0xdeadbeef, 87600}};
    philox4x32_key_t key = {{127, 0xdecafbad}};
    philox4x32_ctr_t ctr = {{0, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};
    (void)argc; (void)argv; /* unused */
    printf( "The first few randoms with key %llx %llx\n",
	   (unsigned long long)key.v[0], (unsigned long long)key.v[1]);
    for(i=0; i<10; ++i){
        ctr.v[0] = i;
	{
          philox4x32_ctr_t rand =  philox4x32(ctr, key);
          //printf("ctr: %llx %llx threefry2x64(20, ctr, key): %llx %llx\n",
          //       (unsigned long long)ctr.v[0], (unsigned long long)ctr.v[1],
          //       (unsigned long long)rand.v[0], (unsigned long long)rand.v[1]);
          printf("%f\n", ((double) rand.v[0] ) / 2 / 2147483647);
	}
    }
    return 0;
}

#ifndef _POLYMORPHIC_ELEMENTAL_H_
#define _POLYMORPHIC_ELEMENTAL_H_

#include "elemental.hpp"

namespace elemental {
namespace runtime {

enum DataType {
  SINGLE,
  SINGLE_COMPLEX,
  DOUBLE,
  DOUBLE_COMPLEX
};

class RuntimeDistMatrix {
public:
  const DataType dtype_;
  const elemental::Distribution colDist_;
  const elemental::Distribution rowDist_;
  RuntimeDistMatrix(DataType dtype,
                    elemental::Distribution colDist,
                    elemental::Distribution rowDist) :
    dtype_(dtype),
    colDist_(colDist),
    rowDist_(rowDist) {
  }
    
  virtual ~RuntimeDistMatrix();
};


RuntimeDistMatrix* CreateDistMatrix(DataType dtype,
                                    elemental::Distribution colDist,
                                    elemental::Distribution rowDist,
                                    int height,
                                    int width,
                                    elemental::Grid *grid);

void Gemm(std::complex<double> alpha,
          RuntimeDistMatrix *A,
          RuntimeDistMatrix *B,
          std::complex<double> beta,
          RuntimeDistMatrix *C);

}
}

#endif

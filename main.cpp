#include <iostream>

#include "mat.hpp"

using namespace std;

int main() {
  Mat<> a(3, 4);
  Mat<> b = {1, 2, 3, 4};
  Mat<> c = {{1, 2, 3},
             {4, 5, 6}};

  cout << c << endl;
  return 0;
}

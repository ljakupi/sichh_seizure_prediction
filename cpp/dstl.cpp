/// This code implements the algorithm described in 
/// Iasemidis, L. D., Principe, J. C., & Sackellares, 
/// J. C. (2000). Measurement and quantification of 
/// spatiotemporal dynamics of human epileptic seizures.
/// Nonlinear biomedical signal processing, 2, 294-318.

#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/numpy/ufunc.hpp> 
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/extract.hpp>

namespace np = boost::python::numpy;
namespace p = boost::python;
typedef std::vector< std::vector <double > > matrix_d;

struct Parameters{
  Parameters(double V=0.1, double b=0.05, double c=0.1, size_t lag=6, size_t d=7, size_t dt=12)
    : m_V(V),m_b(b),m_c(c),m_lag(lag),m_d(d),m_dt(12)
    {
      m_IDIST3=(d-1) * lag;
    }
  void set_delta(const double delta){ m_delta = delta;}
  void set_V(const double V){ m_V = V;}
  void set_c(const double c){ m_c = c;}
  double m_delta = 0.;
  double m_V = 0.1;
  double m_b;
  double m_c;
  size_t m_lag;
  size_t m_d;
  size_t m_dt;
  size_t m_IDIST3;
};


bool update_Vc(double &V, double &c);
np::ndarray stlm(const np::ndarray &embedded_signals);
std::vector<matrix_d> get_vector(const np::ndarray &array);
std::vector<double> unit_vector(const std::vector<double> &v);
double dot(const std::vector<double> &v_1, const std::vector<double> &v_2);
std::vector<matrix_d> row_normalize(const std::vector<matrix_d> &v_matrix);
double nangle(const std::vector<double> &v1, const std::vector<double> &v2);
double distance(const std::vector<double> &vec_1, const std::vector<double> &vec_2);
double get_delta(const matrix_d &embedded_signal, const size_t lag, const size_t IDIST3, const size_t t);
size_t get_transverse(const matrix_d &embedded_signal, const matrix_d &norm_embedded_signal, size_t t, size_t dt);
std::pair<double,double> get_perturbations(const matrix_d &embedded_signal, const matrix_d &norm_embedded_signal, const size_t  t, Parameters &parameters);
bool check_transverse(const matrix_d &embedded_signal, const matrix_d &norm_embedded_signal, const size_t t_1, const size_t t_2, Parameters &parameters);

std::vector<matrix_d> get_vector(const np::ndarray &array)
{
  /// Return a std::vector built from a python numpy array
  std::vector<matrix_d> vec;
  size_t arr_size = array.shape(0);
  size_t md_size = array.shape(1);
  for(size_t i(0); i<arr_size ; ++i) {
    matrix_d tmp_md;
    for(size_t j(0); j<md_size ; ++j) {
     tmp_md.push_back(std::vector< double >( p::stl_input_iterator< double >( array[i][j] ),
                    p::stl_input_iterator< double >( ) ));
    }
    vec.push_back(tmp_md);
  }
  return vec;
}

double distance(const std::vector<double> &vec_1, const std::vector<double> &vec_2)
{
  /// return the distance between two vectors
  double dist=0;
  for(size_t i(0);i<vec_1.size();++i) {
    dist += (vec_1[i]-vec_2[i])*(vec_1[i]-vec_2[i]);
  }
  return std::sqrt(dist);
}

std::vector<double> unit_vector(const std::vector<double> &v)
{
   /// builds a unit vector, not used here as it is a costly operation.
   /// A normalized matrix is built beforehand
  std::vector<double> v_u(v);
  double norm(0);
  for(const auto &i : v) {
    norm +=i*i;
  }
  norm=std::sqrt(norm);
  for(auto &i : v_u) {
    i/=norm;
  }
  return v_u;
}

double dot(const std::vector<double> &v_1, const std::vector<double> &v_2)
{
  /// Dot product between two vectors
  double product=0;
  for(size_t i(0);i<v_1.size();++i) {
    product+=v_1[i]*v_2[i];
  }
  return product;
}

double nangle(const std::vector<double> &v1, const std::vector<double> &v2)
{
    /// Angle between two normalized vectors
    double angle=std::fabs(std::acos( dot(v1, v2)));
    if(std::isfinite(angle)){
      return angle;
    }
    else{ return 1; }
}

double get_delta(const matrix_d &embedded_signal, const size_t lag, const size_t IDIST3, const size_t t)
{
    /// Get delta according to paper
    double delta(0);
    for( size_t delta_t(lag); delta_t<IDIST3; ++delta_t) {
      if( t+delta_t < embedded_signal.size()) {
        delta = std::max(delta, distance(embedded_signal[t],embedded_signal[t+delta_t]));
      }
      if( t > delta_t) {
        delta = std::max(delta, distance(embedded_signal[t],embedded_signal[t-delta_t]));
      }
    }
    return delta;
}


bool update_Vc(double &V, double &c)
{
  /// return true if parameters are updated
  /// return false if update is no longer possible
    if(c<0.51 && c>0.49 && V>0.99 && V<1.01) {
      return false;
    }
    if(c<0.51 && c>0.49) {
      V = std::min(1.,V*2.);
      c = 0.1;
    }
    else {
      c+=0.1;
    }
    return true;
}

bool check_transverse(const matrix_d &embedded_signal, const matrix_d &norm_embedded_signal, const size_t t_1, const size_t t_2, Parameters &parameters)
{
  /// return true if transverse meets the condition
  /// return false if not
  //~ double Vij = 0.1;
  double Vij = nangle(norm_embedded_signal[t_1], norm_embedded_signal[t_2]);
  if(Vij<parameters.m_V){
    double dist =  distance(embedded_signal[t_1], embedded_signal[t_2]);
    if(dist > parameters.m_b* parameters.m_delta && dist < parameters.m_c*parameters.m_delta) {
      return true;
    }
  }
  return false;
}
size_t get_transverse(const matrix_d &embedded_signal, const matrix_d &norm_embedded_signal, size_t t, Parameters &parameters)
{
  /// get vector transverse in the phase space at time t
  parameters.set_c(0.1);
  parameters.set_V(0.1);
  double  delta = get_delta(embedded_signal,parameters.m_lag, parameters.m_IDIST3, t);
  parameters.set_delta(delta);
  while(true) {
    for(size_t delta_t(parameters.m_IDIST3); delta_t < embedded_signal.size(); ++delta_t) {
      if (t+delta_t+parameters.m_dt < embedded_signal.size() && check_transverse(embedded_signal,norm_embedded_signal,t,t+delta_t, parameters)) {
        return t+delta_t;
      }
      if (t > delta_t && check_transverse(embedded_signal,norm_embedded_signal,t,t-delta_t, parameters)) {
        return t-delta_t;
      }
    }
    if(!update_Vc(parameters.m_V,parameters.m_c)) {
      return -1;
    }
  }
}

std::pair<double,double> get_perturbations(const matrix_d &embedded_signal, const matrix_d &norm_embedded_signal, const size_t  t, Parameters &parameters)
{
    /// Compute perturbations between t and dt
    size_t j = get_transverse(embedded_signal, norm_embedded_signal, t, parameters);
    if( j == -1) {
      return std::pair<double,double> (1,1);
    }
    double d_0 = distance(embedded_signal[t],embedded_signal[j]);
    double d_dt = distance(embedded_signal[t+parameters.m_dt],embedded_signal[j+parameters.m_dt]);
    return std::pair<double,double> (d_0,d_dt);
}

std::vector<matrix_d> row_normalize(const std::vector<matrix_d> &v_matrix)
{
  /// return a row normalized matrix used for faster angle computation
  size_t N_matrix = v_matrix.size();
  size_t rows = v_matrix[0].size();
  size_t cols = v_matrix[0][0].size();
  
  auto norm_matrix = v_matrix;
  for(size_t n(0) ; n<N_matrix ; ++n) {
    for(size_t row(0); row<rows ; ++row) {
      double norm=0;
      for(const auto elem : norm_matrix[n][row]) {  
        norm += elem*elem;
      }
      for(auto &elem : norm_matrix[n][row]) {  
        elem*=1./std::sqrt(norm);
      }
    }
  }
  return norm_matrix;
}

np::ndarray stlm(const np::ndarray &embedded_signals)
{
  /// Compute lyapunov exponent and return DSTL matrix
  Parameters parameters;
  std::vector<matrix_d> vec_signals = get_vector(embedded_signals);
  std::vector<matrix_d> norm_vec_signals = row_normalize(vec_signals);
  int N_channels = vec_signals.size();
  int N = vec_signals[0].size();
  std::vector<double> STLm = std::vector<double>(N_channels, 0);
  for(size_t channel(1);channel < N_channels; ++channel) {
    std::cout << channel+1 << "/" << N_channels << std::endl;
    if(N>parameters.m_dt){
      for(int t(0); t<N-parameters.m_dt;++t) {
        auto d = get_perturbations(vec_signals[channel], norm_vec_signals[channel], t, parameters);
        STLm[channel] += std::log2(d.first/d.second);
      }
    }
    STLm[channel] *= 1./((N-parameters.m_dt)*parameters.m_dt);
  }
  std::vector<double> DSTL(N_channels*(N_channels-1)/2.,0);
  size_t idx(0);
  for (size_t channel_a(0); channel_a < N_channels-1; ++channel_a) {
    for (size_t channel_b(channel_a+1); channel_b < N_channels; ++channel_b) {
      DSTL[idx] = std::fabs(STLm[channel_a]-STLm[channel_b]);
      ++idx;
    }
  }
  
  Py_intptr_t shape[1] = { static_cast<long int>(DSTL.size()) };
  np::ndarray DSTL_np = np::zeros(1, shape, np::dtype::get_builtin<double>());
  std::copy(DSTL.begin(), DSTL.end(), reinterpret_cast<double*>(DSTL_np.get_data()));
  return DSTL_np;
}


BOOST_PYTHON_MODULE(dstl)
{
    Py_Initialize();
    np::initialize();
    p::def("stlm", stlm);
}

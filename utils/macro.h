#pragma once

#ifdef _WINDOWS
#  define POSTINF_EXP __declspec (dllexport)
#  define POSTINF_IMP __declspec (dllimport)
#  ifdef POSTINF_EXPORTS
#    define POSTINF_DECL __declspec (dllexport)
#  else
#    define POSTINF_DECL __declspec (dllimport)
#  endif
#  pragma warning( disable: 4251 )
#  pragma warning( disable: 4275 )
#  if (_MSC_VER >= 1400) // vc8
#    pragma warning(disable : 4996) //_CRT_SECURE_NO_DEPRECATE
#  endif
#else
#  if __GNUC__ >= 4
#  define POSTINF_EXP  __attribute__ ((visibility ("default")))
#  define POSTINF_DECL __attribute__ ((visibility ("default")))
#  define POSTINF_IMP __attribute__ ((visibility ("default")))
#  else
#  define POSTINF_EXP
#  define POSTINF_DECL
#  define POSTINF_IMP
#  endif
#endif

#  define POSTINF_LOCAL  

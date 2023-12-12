
#include "utils/os.hpp"
#include <fstream>
#include <iostream>
#include "spdlog/async.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#ifdef _WIN32
#include <direct.h>		//for mkdir rmdir
#include <io.h>			//for access
#elif __linux__
#include <unistd.h>		//for mkdir rmdir
#include <sys/stat.h>	//for access
#include <dirent.h>		//for DIR remove
#endif
 
#ifdef _WIN32
#define ACCESS _access
#define MKDIR(a) _mkdir((a))
#define RMDIR(a) _rmdir((a))
#elif __linux__
#define ACCESS access
#define MKDIR(a) mkdir((a),0755)
#define RMDIR(a) rmdir((a))
#endif

string replace_str(string str, const string& to_replaced, const string& newchars)
{
    for(string::size_type pos(0); pos != string::npos; pos += newchars.length())
    {
        pos = str.find(to_replaced,pos);
        if(pos!=string::npos)
           str.replace(pos,to_replaced.length(),newchars);
        else
            break;
    }
    return   str;
}



namespace os {

std::shared_ptr<spdlog::logger> get_logger() {
  std::string name = PROJECT_NAME;
  auto logger = spdlog::get(name);
  if (!logger) logger = spdlog::stdout_color_mt(name);
  return logger;
}
void set_log_level(LogLevel level) {
  get_logger()->set_level(spdlog::level::level_enum(level));
#ifdef WITH_PIPELINE
  // geods::set_log_level(geods::LogLevel(level));
#endif
}

string path_join(string dir1, string dir2) {
    #ifdef _WIN32
      return dir1 + "\\" + dir2;
    #elif __linux__
      return dir1 + "/" + dir2;
    #endif
}

bool path_isfile(string path) {
  fstream _file;
  _file.open(path.c_str(), ios::in);
  if (!_file)
    return false;
  else
    return true;
}

string path_basename(string path) {
    #ifdef _WIN32
      int pos1=path.find_last_of('\\');  
      int pos2=path.find_last_of('.');  
      string s1(path.substr(pos1+1,pos2));  
      return s1;
    #elif __linux__
      int pos1=path.find_last_of('/');  
      int pos2=path.find_last_of('.');  
      string s1(path.substr(pos1+1,pos2));  
      return s1;
    #endif
}

string path_suffix(string path) {
    #ifdef _WIN32
      int pos2=path.find_last_of('.');  
      string s1(path.substr(pos2+1));  
      return s1;
    #elif __linux__
      int pos2=path.find_last_of('.');  
      string s1(path.substr(pos2+1)); 
      return s1;
    #endif
}

vector<string> path_splitext(string path) {
    int pos1 = path.find_last_of('.');
    string s1(path.substr(0, pos1));
    string s2(path.substr(pos1 + 1));
    vector<string> res{s1, s2};
    return res;
}

bool makedirs(const std::string& strPath) {
  int i = 0;
  int nDirLen = strPath.length();
  if (nDirLen <= 0) return false;
  char* pDirTemp = new char[nDirLen + 4];
  strPath.copy(pDirTemp, nDirLen + 1, 0);  // +1 to copy '\0'
  pDirTemp[nDirLen] = '\0';
  //在末尾加'/'
  if (pDirTemp[nDirLen - 1] != '\\' && pDirTemp[nDirLen - 1] != '/') {
    pDirTemp[nDirLen] = '/';
    pDirTemp[nDirLen + 1] = '\0';
    nDirLen++;
  }
  // 创建目录
  for (i = 0; i < nDirLen; i++) {
    if (pDirTemp[i] == '\\' || pDirTemp[i] == '/') {
      pDirTemp[i] = '\0';  //截断后面的子目录，逐级查看目录是否存在，若不存在则创建
      //如果不存在,创建
      int statu;
      statu = ACCESS(pDirTemp, 0);
      if (statu != 0)  //可能存在同名文件导致没有创建
      {
        statu = MKDIR(pDirTemp);
        if (statu != 0)  //可能上级不是文件夹而是同名文件导致创建失败
        {
          return false;
        }
      }
      //支持linux,将所有\换成/
      pDirTemp[i] = '/';
    }
  }
  delete[] pDirTemp;
  return true;
}

bool removedirs(const std::string& path) {
  std::string strPath = path;
#ifdef _WIN32
  struct _finddata_t fb;  //查找相同属性文件的存储结构体
  //制作用于正则化路径
  if (strPath.at(strPath.length() - 1) != '\\' || strPath.at(strPath.length() - 1) != '/') strPath.append("\\");
  std::string findPath = strPath + "*";
  intptr_t handle;  //用long类型会报错
  handle = _findfirst(findPath.c_str(), &fb);
  //找到第一个匹配的文件
  if (handle != -1L) {
    std::string pathTemp;
    do  //循环找到的文件
    {
      //系统有个系统文件，名为“..”和“.”,对它不做处理
      if (strcmp(fb.name, "..") != 0 && strcmp(fb.name, ".") != 0)  //对系统隐藏文件的处理标记
      {
        //制作完整路径
        pathTemp.clear();
        pathTemp = strPath + std::string(fb.name);
        //属性值为16，则说明是文件夹，迭代
        if (fb.attrib == _A_SUBDIR)  //_A_SUBDIR=16
        {
          removedirs(pathTemp.c_str());
        }
        //非文件夹的文件，直接删除。对文件属性值的情况没做详细调查，可能还有其他情况。
        else {
          remove(pathTemp.c_str());
        }
      }
    } while (0 == _findnext(handle, &fb));  //判断放前面会失去第一个搜索的结果
    //关闭文件夹，只有关闭了才能删除。找这个函数找了很久，标准c中用的是closedir
    //经验介绍：一般产生Handle的函数执行后，都要进行关闭的动作。
    _findclose(handle);
  }
  //移除文件夹
  return RMDIR(strPath.c_str()) == 0 ? true : false;

#elif __linux__
  if (strPath.at(strPath.length() - 1) != '\\' || strPath.at(strPath.length() - 1) != '/') strPath.append("/");
  DIR *d = opendir(strPath.c_str());  //打开这个目录
  if (d != NULL) {
    struct dirent *dt = NULL;
    while (dt = readdir(d))  //逐个读取目录中的文件到dt
    {
      //系统有个系统文件，名为“..”和“.”,对它不做处理
      if (strcmp(dt->d_name, "..") != 0 && strcmp(dt->d_name, ".") != 0)  //判断是否为系统隐藏文件
      {
        struct stat st;        //文件的信息
        std::string fileName;  //文件夹中的文件名
        fileName = strPath + std::string(dt->d_name);
        stat(fileName.c_str(), &st);
        if (S_ISDIR(st.st_mode)) {
          removedirs(fileName);
        } else {
          remove(fileName.c_str());
        }
      }
    }
    closedir(d);
  }
  return rmdir(strPath.c_str()) == 0 ? true : false;
#endif
}

}  // namespace os
#pragma once

#include <stdio.h>
#include <string>
#include <string.h>
#include <vector>
#include "spdlog.h"
#include"utils/macro.h"

using namespace std;

POSTINF_DECL string replace_str(string str, const string& to_replaced, const string& newchars);

namespace os {
/** @brief 日志等级
 */
enum class LogLevel {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERR = 4,
  CRITICAL = 5,
  OFF = 6
};
POSTINF_DECL std::shared_ptr<spdlog::logger> get_logger();
POSTINF_DECL void set_log_level(LogLevel level);

POSTINF_DECL string path_join(string dir1, string dir2) ;
POSTINF_DECL bool path_isfile(string path) ;

POSTINF_DECL string path_basename(string path);

POSTINF_DECL string path_suffix(string path);

POSTINF_DECL vector<string> path_splitext(string path);
POSTINF_DECL bool makedirs(const std::string& strPath);

POSTINF_DECL bool removedirs(const std::string& path);

}  // namespace os
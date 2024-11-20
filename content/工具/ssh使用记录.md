---
draft: false
tags:
---

ssh方便又常用，应该好好学一学。
草....好像有漏洞哇 [OpenSSH服务器漏洞的核心技术解释 - regreSSHion - CVE-2024-6387\_哔哩哔哩\_bilibili](https://www.bilibili.com/video/BV11U411U78q)

[[各种网络问题#ssh相关]]

基本操作 [SSH原理与运用（二）：远程操作与端口转发 - 阮一峰的网络日志](https://www.ruanyifeng.com/blog/2011/12/ssh_port_forwarding.html)

## ssh端口转发


[A Visual Guide to SSH Tunnels: Local and Remote Port Forwarding](https://iximiuz.com/en/posts/ssh-tunnels/)
[Port Forwarding (SSH, The Secure Shell: The Definitive Guide)](https://docstore.mik.ua/orelly/networking_2ndEd/ssh/ch09_02.htm)

#### `-L` 本地端口转发（Local Port Forwarding） 
使用 `-L` 选项时，你是在告诉SSH客户端把**对某个本地主机端口的访问**转发到远程

```
-L [bind_address:]port:host:hostport
```
其中：
- `[bind_address:]` 是可选的，表示在**本地机器**上监听的IP地址，默认为回环接口。 
- `port` 是本地机器上你要监听的端口号。
- `host` 是远程主机上的主机名或IP地址，通常为 `localhost` 或 `127.0.0.1`。
- `hostport` 是远程主机上你要转发到的端口号

#### `-R` 远程端口转发（Remote Port Forwarding）
使用 `-R` 选项时，你是在告诉SSH客户端，把**对某个远程主机端口的访问**转发到本地

```
-R [bind_address:]port:host:hostport
```
其中：
- `[bind_address:]` 是可选的，表示在**远程主机**上监听的IP地址，默认为回环接口。
	- If the _bind_address_ is not specified, the default is to only bind to loopback addresses. If the _bind_address_ is '*' or an empty string, then the forwarding is requested to listen on all interfaces.
- `port` 是远程主机上你要监听的端口号。
- `host` 是本地机器上的主机名或IP地址。
- `hostport` 是本地机器上你要转发到的端口号。

##### 远程转发用例
如果和远程服务器只有ssh连接，想要将本地服务（比如翻墙代理）暴露给远程服务器，可以这样：
```
 ssh -R localhost:7890:localhost:7890 username@remote_host -p remote_port
```
反过来如果想将远程服务暴露给本地，只需要添加`-L 本地端口:远程端口`即可。（不过一般vscode会自动帮你把远程服务器的端口转发过来）

运行后会打开一个交互式会话，只要会话没中断，端口转发就在正常运行。添加`-f`可后台运行。

也可以在客户端配置文件里写入设置，这样每建立一个连接就会转发一次。连接断开即结束。

## 配置文件

### 服务器端 /etc/ssh/sshd_config
https://linux.die.net/man/5/sshd_config
#### 权限设置



### 客户端  ~/.ssh/config or /etc/ssh/ssh_config
https://linux.die.net/man/5/ssh_config
```
Host *
  ServerAliveInterval 15
  ServerAliveCountMax 6
  RemoteForward localhost:7890 localhost:17890
```

ssh连接远程服务器时，会根据远程主机名匹配这里的配置文件。这里Host \*是通配符，就应用到所有远程主机。上面给每个主机设置了超时和远程端口转发。

vscode通过ssh连接远程也是一样。查看vscode的输出窗口（相当于控制台日志）可以发现`Running script with connection command: "C:\Windows\System32\OpenSSH\ssh.exe" -T -D 61448 "py2-host" bash`，也是通过ssh，因此也读取了这个配置文件，也会进行自动转发。

一个小问题是，如果我和同一台主机建立了多个ssh连接会怎样？会重复转发吗？会转发给谁呢？哎，可能和端口占用有关？先来先得？转发失败会显示吗？


sshd
https://linux.die.net/man/8/sshd

ssh 
https://linux.die.net/man/1/ssh




### 连接保活

服务器端 /etc/ssh/sshd_config
```
ClientAliveInterval 30
ClientAliveCountMax 3
```

客户端  ~/.ssh/config or /etc/ssh/ssh_config
```
ServerAliveInterval 30
ServerAliveCountMax 3
```
按我的理解，上面的配置是说，客户端每30s发送一个包，如果连续三个包都没有回应，客户端就认为服务器端已经关闭了。这样客户端就需要等待30s\*3=90s的时间才能关闭连接。如果`ServerAliveInterval 0`，就不在应用层做任何处理，传输层断开就马上断开了。
服务器也是一样，上面的设置让服务器允许客户端连续90s没有响应。
[sshd_config(5) - Linux man page](https://linux.die.net/man/5/sshd_config)

上面说的ssh是应用层，传输层也有自己的处理方法
tcp连接关闭的情况：
1. 对方/自己主动关闭（就是四次挥手） 
2. 超时，对方太长时间没回应。这里有两种可能，一是对方连着但是没东西发送，二是对方意外断开了。检测有两种思路：1. 默不作声的等对方发包，超时了就断开 2. 自己定时发包(TCPKeepAlive)，如果对方连续几个包没回应就断开


登录过程：
[SSH协议握手核心过程\_哔哩哔哩\_bilibili](https://www.bilibili.com/video/BV13P4y1o76u)




 #todo 
[ip - ssh -R binds to 127.0.0.1 only on remote - Server Fault](https://serverfault.com/questions/997124/ssh-r-binds-to-127-0-0-1-only-on-remote)

 
[SSH登录Linux长时间不操作就会自动断开问题-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/1788071)
[linux - SSH Timeouts with ClientAliveInterval and ClientAliveCountMax - Server Fault](https://serverfault.com/questions/1080684/ssh-timeouts-with-clientaliveinterval-and-clientalivecountmax)

[\[原\]Linux ssh远程连接断开问题处理办法-阿里云开发者社区](https://developer.aliyun.com/article/572373)




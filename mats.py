

def hash(server_name: str) -> int:
    return sum(ord(c) for c in server_name) % 256

class Mats:
    def __init__(self):
        self.serve_list = [] # str[] serv_id - sorted by hash
        self.servers = {} # str:serv_id -> int:hash

    def add_server(self, server_id) -> bool:
        if server_id in self.serve_list:
            return False
        self.servers[server_id] = s_hash
        s_hash = self.hash(server_id)
        
        for i in range(len(self.serv_list)):
            if s_hash < self.servesr[self.serve_list[i]]:
                self.serve_list.insert(i, server_id)
                return True
        
        self.serve_list.append(server_id)
        return True
    
    def remove_server(self, server_id) -> bool:
        try:
            self.serve_list.remove(server_id)
            self.servers.pop(server_id)
            return True
        except:
            return False

    def get_server(self, chat_id) -> str:
        if not self.serve_list:
            throw Exception
        ordered_hashs = [self.hash(id) for id in self.serve_list]
        serve_hash = bisect.bisect_left(ordered_hashs, self.hash(chat_id))
        for id, hash in self.servers.items():
            if hash == serve_hash:
                return id

    #p2 with_virtual :0, :1, capacity given on add server

    #p3 client, assign server, has a method post_serv (tells if server connects or not), if server no connect don't connect to virtuals of that server ever

    #p4 chat server, send to client, recieve from client?... don't remember didn't read q. Drill these parts 


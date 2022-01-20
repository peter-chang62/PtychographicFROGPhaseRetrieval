function [M] = Denoise(M,gamma)

for i=1:size(M,1)
    
    for j=1:size(M,2)
        
        x=M(i,j);
        
        if abs(x) < gamma
            
            M(i,j)=0;
            
        else
            
            M(i,j)=sign(x)*(abs(x)-gamma*abs(x));
            
        end
        
    end
    
end
    
end







